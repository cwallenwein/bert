from torch import nn
import torch
from model.util import init_xavier
from model.bert.config import BertConfig
from model.attention import MultiHeadAttention
from model.activation_functions import GatedLinearUnit
from model.positional_information import SinusoidalPositionalEmbeddings, ScaledSinusoidalPositionalEmbeddings

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModel(nn.Module):
    config: BertConfig

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config: BertConfig = config
        self.embedding = BertEmbedding(config)
        self.encoder = nn.ModuleList([
            BertEncoderLayer(config) for _ in range(config.n_layers)
        ])
        if self.config.add_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                normalized_shape=config.d_model, eps=1e-6
            )

    def forward(
        self,
        input_ids: Float[Tensor, "batch sequence_length"],
        token_type_ids: Float[Tensor, "batch sequence_length"],
        attention_mask: Float[Tensor, "batch sequence_length"],
        **kwargs
    ):
        # TODO: define dimensions of return type
        # TODO: define default token_type_ids and attention_mask

        x = self.embedding(
            input_ids=input_ids,
            segment_ids=token_type_ids
        )

        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)

        if self.config.add_final_layer_norm:
            x = self.final_layer_norm(x)

        return x


class BertEmbedding(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.positional_information_type = config.positional_information_type
        self.token_embedding_matrix = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )

        if config.with_next_sentence_prediction:
            self.segment_embedding_matrix = nn.Embedding(
                num_embeddings=config.n_segments,
                embedding_dim=config.d_model
            )

        if config.positional_information_type == "learned":
            self.positional_information = nn.Embedding(
                num_embeddings=config.context_length,
                embedding_dim=config.d_model,
            )
        elif config.positional_information_type == "sinusoidal":
            self.positional_information = SinusoidalPositionalEmbeddings(
                num_embeddings=config.context_length,
                embedding_dim=config.d_model,
            )
        elif config.positional_information_type == "scaled_sinusoidal":
            self.positional_information = ScaledSinusoidalPositionalEmbeddings(
                num_embeddings=config.context_length,
                embedding_dim=config.d_model,
            )

        self.embedding_dropout = nn.Dropout(config.p_embedding_dropout)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.with_next_sentence_prediction = config.with_next_sentence_prediction

        self._init_weights()

    def forward(
        self,
        input_ids: Float[Tensor, "batch sequence_length"],
        segment_ids: Float[Tensor, "batch sequence_length"],
    ):
        sequence_length = input_ids.size(1)
        token_position = torch.arange(sequence_length, device=input_ids.device)

        token_embeddings = self.token_embedding_matrix(input_ids)
        positional_information = self.positional_information(token_position)
        x = token_embeddings + positional_information

        if self.with_next_sentence_prediction:
            segment_embeddings = self.segment_embedding_matrix(segment_ids)
            x += segment_embeddings

        x = self.embedding_dropout(x)
        x = self.layer_norm(x)
        return x

    def _init_weights(self):
        init_xavier(embedding=self.token_embedding_matrix)
        if self.with_next_sentence_prediction:
            init_xavier(embedding=self.segment_embedding_matrix)
        if isinstance(self.positional_information, nn.Embedding):
            init_xavier(embedding=self.positional_information)


class BertEncoderLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU
        }

        if config.feed_forward_activation in activations.keys():
            self.feed_forward = nn.Sequential(
                nn.Linear(config.d_model, config.feed_forward_intermediate_size, bias=config.feed_forward_bias),
                activations[config.feed_forward_activation](),
                nn.Linear(config.feed_forward_intermediate_size, config.d_model, bias=config.feed_forward_bias),
            )
        elif config.feed_forward_activation == "glu":
            self.feed_forward = nn.Sequential(
                nn.Linear(config.d_model, config.feed_forward_intermediate_size, bias=config.feed_forward_bias),
                GatedLinearUnit(),
                nn.Linear(config.feed_forward_intermediate_size // 2, config.d_model, bias=config.feed_forward_bias),
            )
        self.multi_head_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            bias=config.attention_bias,
            attention_implementation=config.attention_implementation
        )

        self.layer_norm1 = nn.LayerNorm(
            normalized_shape=config.d_model, eps=1e-6
        )
        self.layer_norm2 = nn.LayerNorm(
            normalized_shape=config.d_model, eps=1e-6
        )

        self.feed_forward_dropout = nn.Dropout(config.p_feed_forward_dropout)
        self.attention_dropout = nn.Dropout(config.p_attention_dropout)
        self._init_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Float[Tensor, "batch seq_len"]
    ):
        # TODO: define return type
        attention_mask = attention_mask.bool()

        if self.config.layer_norm == "pre":
            x = self.layer_norm1(x)

        x = x + self.attention_dropout(
            self.multi_head_attention(x, attention_mask=attention_mask)
        )

        if self.config.layer_norm == "post":
            x = self.layer_norm1(x)
        elif self.config.layer_norm == "pre":
            x = self.layer_norm2(x)

        x = x + self.feed_forward_dropout(
            self.feed_forward(x)
        )

        if self.config.layer_norm == "post":
            x = self.layer_norm2(x)

        return x

    def _init_weights(self):
        init_xavier(sequential=self.feed_forward)
