from torch import nn
from model.config import BertConfig
from model.attention import MultiHeadAttentionBuilder
from model.embedding import BertEmbedding
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