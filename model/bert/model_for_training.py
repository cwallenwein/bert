from torch import nn
from model.bert.config import BertConfig
from model.bert.model import BertModel

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModelForPretraining(nn.Module):
    # TODO: implement sparse token prediction
    # TODO: implement and test flash attention
    # TODO: add an option for weight tying to the config
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config: BertConfig = config
        self.bert = BertModel(config)
        self.masked_language_modeling_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.with_next_sentence_prediction:
            self.next_sentence_prediction_head = nn.Linear(config.d_model, 1, bias=False)

    def forward(
        self,
        input_ids: Float[Tensor, "batch sequence_length"],
        attention_mask: Float[Tensor, "batch sequence_length"],
        token_type_ids: Float[Tensor, "batch sequence_length"] = None,
        **kwargs
    ):

        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        mlm_output = self.masked_language_modeling_head(bert_output)
        if self.config.with_next_sentence_prediction:
            nsp_output = self.next_sentence_prediction_head(
                bert_output[..., 0, :]
            ).squeeze(-1)
            return mlm_output, nsp_output
        else:
            return mlm_output, None


class BertModelForSequenceClassification(nn.Module):

    def __init__(self, pretrained_model: BertModel, num_classes: int):
        super().__init__()
        self.config = pretrained_model.config
        self.num_classes = num_classes

        self.bert = pretrained_model
        self.classification_head = nn.Linear(self.config.d_model, num_classes, bias=False)

        # freeze bert
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids: Float[Tensor, "batch sequence_length"],
            attention_mask: Float[Tensor, "batch sequence_length"],
            **kwargs
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attention_mask
        )
        classification_output = self.classification_head(
            bert_output[..., 0, :]
        )
        return classification_output.squeeze(-1)
