import torch
from torch import nn
from model.config import BertConfig
from model.attention import MultiHeadAttentionBuilder, ScaledDotProductAttention
from model.util import init_xavier
from model.gated_linear_unit import GatedLinearUnit2

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModelForPretraining(nn.Module):
    # TODO: add an option for weight tying to the config
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config: BertConfig = config
        self.bert = BertModel(config)
        self.masked_language_modeling_head = nn.Linear(config.d_model, config.vocab_size)
        self.next_sentence_prediction_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids: Float[Tensor, "batch sequence_length"],
        token_type_ids: Float[Tensor, "batch sequence_length"],
        attention_mask: Float[Tensor, "batch sequence_length"],
        **kwargs
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        mlm_output = self.masked_language_modeling_head(bert_output)
        nsp_output = self.next_sentence_prediction_head(
            bert_output[..., 0, :]
        ).squeeze(-1)
        return mlm_output, nsp_output


class BertModel(nn.Module):
    config: BertConfig

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config: BertConfig = config
        self.embedding = BertEmbedding(config)
        self.encoder = nn.ModuleList([
            BertEncoderLayer(config) for _ in range(config.n_layers)
        ])

    def forward(
        self,
        input_ids: Float[Tensor, "batch sequence_length"],
        token_type_ids: Float[Tensor, "batch sequence_length"],
        attention_mask: Float[Tensor, "batch sequence_length"],
        **kwargs
    ):

        x = self.embedding(
            input_ids=input_ids,
            segment_ids=token_type_ids
        )

        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)

        return x


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
                GatedLinearUnit2(),
                nn.Linear(config.feed_forward_intermediate_size // 2, config.d_model, bias=config.feed_forward_bias),
            )
        self.multi_head_attention = MultiHeadAttentionBuilder(config).build()

        self.layer_norm1 = nn.LayerNorm(
            normalized_shape=config.d_model
        )
        self.layer_norm2 = nn.LayerNorm(
            normalized_shape=config.d_model
        )

        self.feed_forward_dropout = nn.Dropout(config.p_feed_forward_dropout)
        self.attention_dropout = nn.Dropout(config.p_attention_dropout)
        self._init_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Float[Tensor, "batch seq_len"]
    ):
        attention_mask = attention_mask.bool()
        x = x + self.attention_dropout(
            self.multi_head_attention(x, attention_mask=attention_mask)
        )
        x = self.layer_norm1(x)
        x = x + self.feed_forward_dropout(
            self.feed_forward(x)
        )
        x = self.layer_norm2(x)
        return x

    def _init_weights(self):
        init_xavier(sequential=self.feed_forward)


class BertEmbedding(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.positional_information_type = config.positional_information_type
        self.token_embedding_matrix = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )

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

        self.layer_norm = nn.LayerNorm(config.d_model)
        self.embedding_dropout = nn.Dropout(config.p_embedding_dropout)

        self._init_weights()

    def forward(
        self,
        input_ids: Float[Tensor, "batch sequence_length"],
        segment_ids: Float[Tensor, "batch sequence_length"],
    ):
        sequence_length = input_ids.size(1)
        token_position = torch.arange(sequence_length, device=input_ids.device)

        token_embeddings = self.token_embedding_matrix(input_ids)
        segment_embeddings = self.segment_embedding_matrix(segment_ids)
        positional_information = self.positional_information(token_position)

        x = token_embeddings + segment_embeddings + positional_information
        x = self.layer_norm(x)
        x = self.embedding_dropout(x)
        return x

    def _init_weights(self):
        init_xavier(embedding=self.token_embedding_matrix)
        init_xavier(embedding=self.segment_embedding_matrix)
        if isinstance(self.positional_information, nn.Embedding):
            init_xavier(embedding=self.positional_information)


class SinusoidalPositionalEmbeddings(nn.Module):
    """
    Based on https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, n: int = 10_000):
        super().__init__()
        positional_encoding = torch.zeros(num_embeddings, embedding_dim)

        positions = torch.arange(num_embeddings).unsqueeze(1)
        frequencies = (
            n ** torch.arange(embedding_dim // 2) / (embedding_dim // 2)
        ).unsqueeze(0)

        values = torch.div(positions, frequencies)

        positional_encoding[:, 0::2] = torch.sin(values)
        positional_encoding[:, 1::2] = torch.cos(values)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self,
        token_position: Float[Tensor, "sequence_length"],
    ):
        seq_length = token_position.size(0)
        return self.positional_encoding[:seq_length, :]


class ScaledSinusoidalPositionalEmbeddings(nn.Module):
    """
    Based on https://proceedings.mlr.press/v162/hua22a/hua22a.pdf
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, n: int = 10_000):
        super().__init__()
        self.scaling_factor = nn.Parameter(torch.ones(1))
        self.sinusoidal_positional_embedding = SinusoidalPositionalEmbeddings(
            embedding_dim=embedding_dim, num_embeddings=num_embeddings, n=n
        )

    def forward(
        self,
        input_ids: Float[Tensor, "sequence_length"],
    ):
        return self.scaling_factor * self.sinusoidal_positional_embedding(input_ids)
