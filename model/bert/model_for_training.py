from torch import nn, optim
from model.bert.config import BertConfig
from model.bert.model import BertModel
import lightning as L

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModelForPretraining(L.LightningModule):
    # TODO: implement sparse token prediction
    # TODO: implement and test flash attention
    # TODO: add an option for weight tying to the config
    def __init__(self, config: BertConfig, learning_rate: float):
        super().__init__()
        self.config: BertConfig = config
        self.bert = BertModel(config)
        self.learning_rate = learning_rate

        # masked language modeling
        self.masked_language_modeling_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # TODO: check if softmax and cross entropy loss is combined?
        self.mlm_loss_fn = nn.CrossEntropyLoss()

        # next sentence prediction
        if config.with_next_sentence_prediction:
            self.next_sentence_prediction_head = nn.Linear(config.d_model, 1, bias=False)
            self.nsp_loss_fn = nn.BCEWithLogitsLoss()

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

    def training_step(self, batch, batch_idx):
        masked_language_modeling_output, next_sentence_prediction_output = self(**batch)

        masked_tokens = batch["masked_tokens_mask"].bool()
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]
        mlm_loss = self.mlm_loss_fn(masked_token_predictions, masked_token_labels)

        if self.config.with_next_sentence_prediction:
            next_sentence_prediction_labels = batch["labels"][..., 0]

            nsp_loss = self.nsp_loss_fn(
                next_sentence_prediction_output, next_sentence_prediction_labels
            )
            return mlm_loss + nsp_loss
        else:
            return mlm_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12
        )
        return optimizer


class BertModelForSequenceClassification(nn.Module):

    def __init__(self, pretrained_model: L.LightningModule, num_classes: int, learning_rate: float):
        super().__init__()
        self.config = pretrained_model.config
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.bert = pretrained_model

        self.classification_head = nn.Linear(self.config.d_model, num_classes, bias=False)
        self.classification_loss_fn = nn.CrossEntropyLoss()

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

    def training_step(self, batch, batch_idx):
        sequence_classification_output = self(**batch)

        classification_loss = self.classification_loss_fn(
            sequence_classification_output, batch["labels"]
        )
        return classification_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12
        )
        return optimizer