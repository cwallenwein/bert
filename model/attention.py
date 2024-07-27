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

        self.query_weight: Float[Tensor, "d_model d_head"] = nn.Parameter(
            torch.randn(config.d_model, config.d_head)
        )
        self.query_bias: Float[Tensor, "d_head"] = nn.Parameter(
            torch.randn(config.d_head)
        )
        self.key_weight: Float[Tensor, "d_model d_head"] = nn.Parameter(
            torch.randn(config.d_model, config.d_head)
        )
        self.key_bias: Float[Tensor, "d_head"] = nn.Parameter(
            torch.randn(config.d_head)
        )
        self.value_weight: Float[Tensor, "d_model d_head"] = nn.Parameter(
            torch.randn(config.d_model, config.d_head)
        )
        self.value_bias: Float[Tensor, "d_head"] = nn.Parameter(
            torch.randn(config.d_head)
        )
        self._init_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Bool[Tensor, "batch seq_len"]
    ):
        query = einsum(
            x, self.query_weight,
            "batch seq_len d_model, d_model d_head -> batch seq_len d_head"
        ) + self.query_bias

        key = einsum(
            x, self.key_weight,
            "batch seq_len d_model, d_model d_head -> batch seq_len d_head"
        ) + self.key_bias

        value = einsum(
            x, self.value_weight,
            "batch seq_len d_model, d_model d_head -> batch seq_len d_head"
        ) + self.value_bias

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
        init_xavier(weights=self.query_weight, bias=self.query_bias)
        init_xavier(weights=self.key_weight, bias=self.key_bias)
        init_xavier(weights=self.value_weight, bias=self.value_bias)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.query_weight = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.key_weight = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.value_weight = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.output_weight = nn.Linear(config.d_model, config.d_model)

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


class MultiHeadAttentionFromScratch(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.query_weight: Float[Tensor, "n_heads d_model d_head"] = nn.Parameter(
            torch.randn(config.n_heads, config.d_model, config.d_head)
        )
        self.query_bias: Float[Tensor, "n_heads d_head"] = nn.Parameter(
            torch.randn(config.n_heads, config.d_head)
        )
        self.key_weight: Float[Tensor, "n_heads d_model d_head"] = nn.Parameter(
            torch.randn(config.n_heads, config.d_model, config.d_head)
        )
        self.key_bias: Float[Tensor, "n_heads d_head"] = nn.Parameter(
            torch.randn(config.n_heads, config.d_head)
        )
        self.value_weight: Float[Tensor, "n_heads d_model d_head"] = nn.Parameter(
            torch.randn(config.n_heads, config.d_model, config.d_head)
        )
        self.value_bias: Float[Tensor, "n_heads d_head"] = nn.Parameter(
            torch.randn(config.n_heads, config.d_head)
        )
        self.output_weight: Float[Tensor, "d_model d_model"] = nn.Parameter(
            torch.randn(config.d_model, config.d_model)
        )
        self.output_bias: Float[Tensor, "d_model"] = nn.Parameter(
            torch.randn(config.d_model)
        )

        self._init_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Bool[Tensor, "batch seq_len"]
    ):

        query = einsum(
            x, self.query_weight,
            "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head"
        )
        query += self.query_bias

        key = einsum(
            x, self.key_weight,
            "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head"
        ) + self.key_bias

        value = einsum(
            x, self.value_weight,
            "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head"
        ) + self.value_bias

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

        output = einsum(
            attention_output, self.output_weight,
            "batch seq_len d_model_in, d_model_in d_model_out -> batch seq_len d_model_out"
        ) + self.output_bias

        return output

    def _init_weights(self):
        init_xavier(weights=self.query_weight, bias=self.query_bias)
        init_xavier(weights=self.key_weight, bias=self.key_bias)
        init_xavier(weights=self.value_weight, bias=self.value_bias)
        init_xavier(weights=self.output_weight, bias=self.output_bias)


class MultiHeadAttentionOptimized(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.query_weight = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.key_weight = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.value_weight = nn.Linear(config.d_model, config.n_heads * config.d_head)
        self.output_weight = nn.Linear(config.d_model, config.d_model)

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
        elif self.config.multi_head_attention_implementation == "from_scratch":
            return MultiHeadAttentionFromScratch(self.config)
        else:
            raise ValueError("Unknown MultiHeadAttention implementation")