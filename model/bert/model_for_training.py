import torch
from torch import nn, optim
from model.bert.config import BertConfig
from model.bert.model import BertModel
import lightning as L
import torchmetrics

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
        self.wandb = None

        # masked language modeling
        self.masked_language_modeling_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # TODO: check if softmax and cross entropy loss is combined?
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.mlm_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=config.vocab_size,
            average="micro"
        ).to(self.device)

        # next sentence prediction
        if config.with_next_sentence_prediction:
            self.next_sentence_prediction_head = nn.Linear(config.d_model, 1, bias=False)
            self.nsp_loss_fn = nn.BCEWithLogitsLoss()
            self.nsp_accuracy = torchmetrics.Accuracy(
                task="binary",
                average="micro"
            ).to(self.device)

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

    def training_step(self, batch, batch_idx, step, micro_step):
        masked_language_modeling_output, next_sentence_prediction_output = self(**batch)

        # only log if first micro_batch to reduce overhead
        if micro_step == 0:
            metrics = self.calculate_metrics(batch, masked_language_modeling_output, next_sentence_prediction_output)
            self.wandb.log(metrics, step=step)

        masked_tokens = batch["masked_tokens_mask"]
        # TODO: don't filter output here
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

    def calculate_metrics(self, batch, masked_language_modeling_output, next_sentence_prediction_output):
        metrics = dict()

        masked_tokens = batch["masked_tokens_mask"]
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]

        # calculate accuracy
        mlm_acc = self.mlm_accuracy(
            masked_token_predictions, masked_token_labels
        )
        # print(mlm_acc, (torch.argmax(masked_token_predictions, dim=1) == masked_token_labels).sum())
        metrics["mlm_acc"] = mlm_acc

        if self.config.with_next_sentence_prediction:
            next_sentence_prediction_labels = batch["labels"][..., 0]

            # calculate accuracy
            metrics["nsp_acc"] = self.nsp_accuracy(
                next_sentence_prediction_output, next_sentence_prediction_labels
            )
        return metrics


class BertModelForSequenceClassification(nn.Module):

    def __init__(self, pretrained_model: L.LightningModule, num_classes: int, learning_rate: float):
        super().__init__()
        self.config = pretrained_model.config
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.bert = pretrained_model

        self.classification_head = nn.Linear(self.config.d_model, num_classes, bias=False)
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.classification_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="micro"
        ).to(self.device)

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

    def training_step(self, batch, batch_idx, step, micro_step):
        sequence_classification_output = self(**batch)

        classification_loss = self.classification_loss_fn(
            sequence_classification_output, batch["labels"]
        )

        # only log if first micro_batch to reduce overhead
        if micro_step == 0:
            metrics = self.calculate_metrics(batch, sequence_classification_output)
            self.wandb.log(metrics, step=step)

        return classification_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12
        )
        return optimizer

    def calculate_metrics(self, batch, sequence_classification_output):
        accuracy = self.classification_accuracy(
            sequence_classification_output, batch["labels"]
        )
        metrics = {"mnli": accuracy}
        return metrics
