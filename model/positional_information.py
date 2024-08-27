import torch
from torch import nn

# Typehints
from jaxtyping import Float
from torch import Tensor


class SinusoidalPositionalEmbeddings(nn.Module):
    """
    Based on https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, n: int = 10_000):
        super().__init__()
        positional_encoding = torch.zeros(num_embeddings, embedding_dim)

        positions = torch.arange(num_embeddings).unsqueeze(1)
        frequencies = (
            n ** (torch.arange(embedding_dim // 2) / (embedding_dim // 2))
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
