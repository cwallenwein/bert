import math
from typing import Annotated

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from bert.model.util import init_xavier

# TODO: Fix attention mask for flash attention
# TODO: add p_attention_dropout
# TODO: merge QKV weights


def scaled_dot_product_attention(
    query: Annotated[Tensor, "batch n_heads seq_len d_head"],
    key: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
    value: Annotated[Tensor, torch.float, "batch n_heads seq_len d_head"],
    attention_mask: Annotated[Tensor, torch.bool, "batch seq_len"],
):
    attention_score = einsum(
        query,
        key,
        """
        batch n_heads query_len d_head,
        batch n_heads key_len d_head
        -> batch n_heads query_len key_len
        """,
    )
    mask = torch.where(attention_mask, 0, float("inf"))
    # add dim for broadcasting (n_heads, query_len)
    mask = mask[:, None, None, :]

    attention_score -= mask
    attention_probability = F.softmax(
        attention_score / math.sqrt(query.size(-1)), dim=-1
    )
    output = einsum(
        attention_probability,
        value,
        """
        batch n_heads query_len key_len,
        batch n_heads seq_len d_head
        -> batch n_heads query_len d_head
        """,
    )

    return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool,
        attention_implementation: str = "default",
    ):
        super().__init__()

        assert attention_implementation in ["default", "pytorch", "flash-attn"]

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bias = bias
        self.attention_implementation = attention_implementation

        # d_model -> n_heads * d_head & d_model == n_heads * d_head
        self.query_weight = nn.Linear(d_model, d_model, bias=bias)
        self.key_weight = nn.Linear(d_model, d_model, bias=bias)
        self.value_weight = nn.Linear(d_model, d_model, bias=bias)
        self.output_weight = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def forward(
        self,
        x: Annotated[Tensor, torch.float, "batch seq_len d_model"],
        attention_mask: Annotated[Tensor, torch.bool, "batch seq_len"],
    ):
        # TODO: define return type
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

        if self.attention_implementation == "default":
            attention_output = scaled_dot_product_attention(
                query, key, value, attention_mask
            )
        elif self.attention_implementation == "pytorch":
            attention_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask[:, None, None, :]
            )
        elif self.attention_implementation == "flash-attn":
            attention_output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                # attn_mask=attention_mask[:, None, None, :]
            ).transpose(1, 2)
        else:
            raise Exception("Unknown multi-head attention implementation")

        attention_output = rearrange(
            attention_output,
            "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)",
        )

        output = self.output_weight(attention_output)

        return output

    def _init_weights(self):
        init_xavier(linear=self.query_weight)
        init_xavier(linear=self.key_weight)
        init_xavier(linear=self.value_weight)
        init_xavier(linear=self.output_weight)
