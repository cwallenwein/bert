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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config
        assert config.n_heads == 1

        self.query_weight = nn.Linear(config.d_model, config.d_head, bias=config.attention_bias)
        self.key_weight = nn.Linear(config.d_model, config.d_head, bias=config.attention_bias)
        self.value_weight = nn.Linear(config.d_model, config.d_head, bias=config.attention_bias)

        self._init_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Bool[Tensor, "batch seq_len"]
    ):
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)

        attention_score = einsum(
            key, query,
            "batch key_len d_head, batch query_len d_head -> batch query_len key_len"
        )
        mask = torch.where(attention_mask, 0, float("inf"))[:, None, :]
        attention_score -= mask
        attention_probability = F.softmax(
            attention_score / math.sqrt(self.config.d_model),
            dim=2
        )

        # d_model == d_head -> correct output dimension
        output = einsum(
            attention_probability, value,
            "batch query_len key_len, batch seq_len d_head -> batch query_len d_head"
        )

        return output

    def _init_weights(self):
        init_xavier(linear=self.query_weight)
        init_xavier(linear=self.key_weight)
        init_xavier(linear=self.value_weight)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.query_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.key_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.value_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.output_weight = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

        self._init_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Bool[Tensor, "batch seq_len"]
    ):
        query = rearrange(
            self.query_weight(x),
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.config.n_heads
        )
        key = rearrange(
            self.key_weight(x),
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.config.n_heads
        )
        value = rearrange(
            self.value_weight(x),
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.config.n_heads
        )

        attention_score = einsum(
            query, key,
            "batch query_len n_heads d_head, batch key_len n_heads d_head -> batch n_heads query_len key_len"
        )

        mask = torch.where(attention_mask, 0, float("inf"))[:, None, None, :]
        attention_score -= mask

        attention_probability = F.softmax(
            attention_score / math.sqrt(self.config.d_head),
            dim=3
        )

        attention_output = einsum(
            attention_probability, value,
            "batch n_heads query_len key_len, batch key_len n_heads d_head -> batch n_heads query_len d_head"
        )

        attention_output = rearrange(
            attention_output,
            "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)"
        )

        output = self.output_weight(attention_output)

        return output

    def _init_weights(self):
        init_xavier(linear=self.query_weight)
        init_xavier(linear=self.key_weight)
        init_xavier(linear=self.value_weight)
        init_xavier(linear=self.output_weight)


class MultiHeadAttentionOptimized(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.query_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.key_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.value_weight = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.attention_bias)
        self.output_weight = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

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

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, None, None, :]
        )

        attention_output = rearrange(
                attention_output,
                "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)"
        )

        output = self.output_weight(attention_output)

        return output

    def _init_weights(self):
        init_xavier(linear=self.query_weight)
        init_xavier(linear=self.key_weight)
        init_xavier(linear=self.value_weight)
        init_xavier(linear=self.output_weight)


class MultiHeadAttentionBuilder:
    def __init__(self, config: BertConfig):
        self.config = config

    def build(self):
        if self.config.multi_head_attention_implementation == "default":
            return MultiHeadAttention(self.config)
        elif self.config.multi_head_attention_implementation == "pytorch":
            return MultiHeadAttentionOptimized(self.config)
        else:
            raise ValueError("Unknown MultiHeadAttention implementation")