import torch
from torch import nn
from model.config import BertConfig
from model.util import init_xavier

# Typehints
from jaxtyping import Float
from torch import Tensor


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

        self.layer_norm = nn.LayerNorm(config.d_model)
        self.embedding_dropout = nn.Dropout(config.p_embedding_dropout)
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

        x = self.layer_norm(x)
        x = self.embedding_dropout(x)
        return x

    def _init_weights(self):
        init_xavier(embedding=self.token_embedding_matrix)
        if self.with_next_sentence_prediction:
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
