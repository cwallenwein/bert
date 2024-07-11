import torch
from torch import nn
from model.config import BertConfig
from model.attention import MultiHeadAttentionFromScratch

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModelForPretraining(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config: BertConfig = config
        self.bert = BertModel(config)
        self.masked_language_modeling_head = nn.Sequential(
            nn.Linear(config.d_model, config.vocab_size),
            nn.Softmax(dim=-1)
        )

        self.next_sentence_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, 2),
            nn.Softmax(dim=-1)
        )

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
        nsp_output = self.next_sentence_prediction_head(bert_output[..., 0, :])
        return mlm_output, nsp_output


class BertModel(nn.Module):
    # TODO: weight initialization
    config: BertConfig

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config: BertConfig = config

        self.encoder = nn.ModuleList([
            BertEncoderLayer(config) for _ in range(config.n_layers)
        ])

        self.token_embedding_matrix = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )

        self.position_embedding_matrix = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.d_model
        )

        self.segment_embedding_matrix = nn.Embedding(
            num_embeddings=config.n_segments,
            embedding_dim=config.d_model
        )

    def forward(
        self,
        input_ids: Float[Tensor, "batch sequence_length"],
        token_type_ids: Float[Tensor, "batch sequence_length"],
        attention_mask: Float[Tensor, "batch sequence_length"],
        **kwargs
    ):
        segment_ids = token_type_ids
        context_length = input_ids.size(1)
        token_position = torch.arange(context_length)

        token_embeddings = self.token_embedding_matrix(input_ids)
        segment_embeddings = self.segment_embedding_matrix(segment_ids)
        position_embeddings = self.position_embedding_matrix(token_position)

        x = token_embeddings + segment_embeddings + position_embeddings

        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)

        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Parameter):
                nn.init.xavier_normal_(module)


class BertEncoderLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU
        }

        self.layer_norm1 = nn.LayerNorm(
            normalized_shape=config.d_model
        )
        self.layer_norm2 = nn.LayerNorm(
            normalized_shape=config.d_model
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.feed_forward_intermediate_size),
            activations[config.feed_forward_activation](),
            nn.Linear(config.feed_forward_intermediate_size, config.d_model),
        )
        self.multi_head_attention = MultiHeadAttentionFromScratch(config)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        attention_mask: Float[Tensor, "batch seq_len"]
    ):
        attention_mask = attention_mask.bool()
        x = x + self.multi_head_attention(x, attention_mask=attention_mask)[0]
        x = self.layer_norm1(x)
        x = x + self.feed_forward(x)
        x = self.layer_norm2(x)
        return x
