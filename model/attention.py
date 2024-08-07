import math
import torch
from torch import nn
from torch.nn import functional as F
from model.config import BertConfig
from model.util import init_xavier

from einops import einsum, rearrange

# type hints
from torch import Tensor
from jaxtyping import Float, Bool


def scaled_dot_product_attention(
    query: Float[Tensor, "batch n_heads seq_len d_head"],
    key: Float[Tensor, "batch n_heads seq_len d_head"],
    value: Float[Tensor, "batch n_heads seq_len d_head"],
    attention_mask: Bool[Tensor, "batch seq_len"]
):
    attention_score = einsum(
        query, key,
        "batch n_heads query_len d_head, batch n_heads key_len d_head -> batch n_heads query_len key_len"
    )
    mask = torch.where(attention_mask, 0, float("inf"))
    # add dim for broadcasting (n_heads, query_len)
    mask = mask[:, None, None, :]

    attention_score -= mask
    attention_probability = F.softmax(
        attention_score / math.sqrt(query.size(-1)),
        dim=-1
    )
    output = einsum(
        attention_probability, value,
        "batch n_heads query_len key_len, batch n_heads seq_len d_head -> batch n_heads query_len d_head"
    )

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.query_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.key_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.value_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.output_weight = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

        if self.config.multi_head_attention_implementation == "default":
            self.scaled_dot_product_attention = ScaledDotProductAttention(config=config)

        self._init_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Bool[Tensor, "batch seq_len"]
    ):
        query = rearrange(
            self.query_weight(x),
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.config.n_heads
        )
        key = rearrange(
            self.key_weight(x),
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.config.n_heads
        )
        value = rearrange(
            self.value_weight(x),
            "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head",
            n_heads=self.config.n_heads
        )

        if self.config.multi_head_attention_implementation == "default":
            attention_output = scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask
            )
        elif self.config.multi_head_attention_implementation == "pytorch":
            attention_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask[:, None, None, :]
            )

        attention_output = rearrange(attention_output, "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)")

        output = self.output_weight(attention_output)

        return output

    def _init_weights(self):
        init_xavier(linear=self.query_weight)
        init_xavier(linear=self.key_weight)
        init_xavier(linear=self.value_weight)
        init_xavier(linear=self.output_weight)