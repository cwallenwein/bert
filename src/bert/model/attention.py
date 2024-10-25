import math
from typing import Annotated, Optional

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from bert.model.util import init_xavier

# torch.fx.wrap("rearrange")
# torch.fx.wrap("einsum")

# TODO: Fix attention mask for flash attention
# TODO: merge QKV weights


try:
    from flash_attn import flash_attn_func  # noqa: F401  #type: ignore
    from flash_attn import flash_attn_varlen_func  # noqa: F401  #type: ignore

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


def scaled_dot_product_attention(
    query: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
    key: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
    value: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
    attention_mask: Optional[
        Annotated[Tensor, torch.bool, "batch query_len key_len"]
    ] = None,
    p_dropout: float = 0.0,
) -> Annotated[Tensor, torch.float, "batch n_heads query_len d_head"]:
    attention_score = einsum(
        query,
        key,
        "batch n_heads query_len d_head, batch n_heads key_len d_head\
-> batch n_heads query_len key_len",
    )

    if attention_mask is not None:
        # add n_heads dimension
        mask = torch.where(attention_mask, 0, float("inf")).unsqueeze(1)
        attention_score -= mask

    attention_probability = F.softmax(
        attention_score / math.sqrt(query.size(-1)),
        dim=-1,
    )
    # randomly drop attention probabilities like in
    #   https://github.com/google-research/bert/blob/master/modeling.py
    # also proposed as DropAttention in https://arxiv.org/pdf/1907.11065
    attention_probability = F.dropout(attention_probability, p=p_dropout)

    output = einsum(
        attention_probability,
        value,
        "batch n_heads query_len key_len, batch n_heads key_len d_head\
-> batch n_heads query_len d_head",
    )

    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool,
        p_dropout: float = 0.0,
        attention_implementation: str = "from-scratch",
    ):
        super().__init__()

        assert attention_implementation in ["from-scratch", "pytorch", "flash-attn"]

        if attention_implementation == "from-scratch":
            self.scaled_dot_product_attention = (
                self.scaled_dot_product_attention_from_scratch
            )
        elif attention_implementation == "pytorch":
            assert hasattr(F, "scaled_dot_product_attention"), (
                "torch.nn.functional.scaled_dot_product_attention is not available. ",
                "Please choose another attention implementation",
            )

            self.scaled_dot_product_attention = (
                self.scaled_dot_product_attention_pytorch
            )
        elif attention_implementation == "flash-attn":
            assert (
                FLASH_ATTENTION_AVAILABLE
            ), "Flash attention not available. Please install flash-attn"
            self.scaled_dot_product_attention = (
                self.scaled_dot_product_attention_flash_attention
            )
        else:
            raise Exception("Unknown multi-head attention implementation")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bias = bias
        self.attention_implementation = attention_implementation
        self.p_dropout = p_dropout

        # d_model -> n_heads * d_head & d_model == n_heads * d_head
        self.query_weight = nn.Linear(d_model, d_model, bias=bias)
        self.key_weight = nn.Linear(d_model, d_model, bias=bias)
        self.value_weight = nn.Linear(d_model, d_model, bias=bias)
        self.output_weight = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def forward(
        self,
        x: Annotated[Tensor, torch.float, "batch seq_len d_model"],
        attention_mask: Optional[Annotated[Tensor, torch.bool, "batch seq_len"]] = None,
    ) -> Annotated[Tensor, torch.float, "batch seq_len d_model"]:
        """
        Args:
            attention_mask: bool
                True -> attend to this position
                False -> don't attend to this position
        """
        query = rearrange(
            self.query_weight(x),
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
        )
        key = rearrange(
            self.key_weight(x),
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
        )
        value = rearrange(
            self.value_weight(x),
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.n_heads,
        )

        p_dropout = self.p_dropout if self.training else 0.0
        attention_output = self.scaled_dot_product_attention(
            query, key, value, attention_mask, p_dropout
        )

        attention_output = rearrange(
            attention_output,
            "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)",
        )

        output = self.output_weight(attention_output)

        return output

    @staticmethod
    def scaled_dot_product_attention_from_scratch(
        query: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        key: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        value: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        attention_mask: Optional[Annotated[Tensor, torch.bool, "batch seq_len"]],
        p_dropout: float = 0.0,
    ) -> Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"]:

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)

        return scaled_dot_product_attention(
            query, key, value, attention_mask, p_dropout=p_dropout
        )

    @staticmethod
    def scaled_dot_product_attention_pytorch(
        query: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        key: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        value: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        attention_mask: Optional[Annotated[Tensor, torch.bool, "batch seq_len"]],
        p_dropout: float = 0.0,
    ) -> Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"]:

        if attention_mask is not None:
            # add empty head dimension and query dimension
            attention_mask = attention_mask[:, None, None, :]

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=p_dropout,
        )

    @staticmethod
    def scaled_dot_product_attention_flash_attention(
        query: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        key: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        value: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
        attention_mask: Optional[Annotated[Tensor, torch.bool, "batch seq_len"]],
        p_dropout: float = 0.0,
    ) -> Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"]:

        if attention_mask is None:
            return flash_attn_func(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=p_dropout,
            ).transpose(1, 2)
        else:
            raise NotImplementedError(
                "Attention with mask not supported for flash-attn yet."
            )

            batch_size = query.size(0)
            query = rearrange(
                query, "batch n_heads seq_len d_head -> (batch seq_len) n_heads d_head"
            )
            key = rearrange(
                key, "batch n_heads seq_len d_head -> (batch seq_len) n_heads d_head"
            )
            value = rearrange(
                value, "batch n_heads seq_len d_head -> (batch seq_len) n_heads d_head"
            )
            tokens_per_sample = attention_mask.sum(dim=1, dtype=torch.int32)
            cu_seq_lens = tokens_per_sample.cumsum(dim=0, dtype=torch.int32)
            max_seqlen = tokens_per_sample.max().item()

            output = flash_attn_varlen_func(
                query,
                key,
                value,
                dropout_p=p_dropout,
                cu_seqlens_q=cu_seq_lens,
                cu_seqlens_k=cu_seq_lens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
            )

            output = rearrange(
                output,
                "(batch seq_len) n_heads d_head -> batch n_heads seq_len d_head",
                batch=batch_size,
            )

            return output

    def _init_weights(self):
        init_xavier(linear=self.query_weight)
        init_xavier(linear=self.key_weight)
        init_xavier(linear=self.value_weight)
        init_xavier(linear=self.output_weight)
