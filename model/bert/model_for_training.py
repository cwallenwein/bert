import torch
from torch import nn, optim
from model.bert.config import BertConfig
from model.bert.model import BertModel
import lightning as L
import torchmetrics
from trainer.scheduler import DynamicWarmupStableDecayScheduler

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModelForPretraining(L.LightningModule):
    # TODO: implement and test flash attention
    # TODO: add an option for weight tying to the config
    def __init__(
        self,
        config: BertConfig,
        learning_rate: float = 1e-4,
        scheduler: str = "OneCycleLR"
    ):
        super().__init__()
        self.config: BertConfig = config
        self.bert = BertModel(config)
        self.learning_rate = learning_rate

        assert scheduler in ["OneCycleLR", "DynamicWarmupStableDecayScheduler"]
        self.scheduler = scheduler

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
        masked_tokens_mask: Float[Tensor, "batch sequence_length"],
        token_type_ids: Float[Tensor, "batch sequence_length"] = None,
        **kwargs
    ):

        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        mlm_output = self.masked_language_modeling_head(bert_output[masked_tokens_mask])
        if self.config.with_next_sentence_prediction:
            nsp_output = self.next_sentence_prediction_head(
                bert_output[..., 0, :]
            ).squeeze(-1)
            return mlm_output, nsp_output
        else:
            return mlm_output, None

    def training_step(self, batch, batch_idx):
        masked_language_modeling_output, next_sentence_prediction_output = self(**batch)

        metrics = self.calculate_metrics(batch, masked_language_modeling_output, next_sentence_prediction_output)

        masked_tokens = batch["masked_tokens_mask"]
        masked_token_predictions = masked_language_modeling_output
        masked_token_labels = batch["labels"][masked_tokens]
        mlm_loss = self.mlm_loss_fn(masked_token_predictions, masked_token_labels)

        if self.config.with_next_sentence_prediction:
            next_sentence_prediction_labels = batch["labels"][..., 0]

            nsp_loss = self.nsp_loss_fn(
                next_sentence_prediction_output, next_sentence_prediction_labels
            )
            return mlm_loss + nsp_loss, metrics
        else:
            return mlm_loss, metrics

    def configure_optimizers(self, training_steps_total):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12
        )

        if self.scheduler == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=training_steps_total
            )
        elif self.scheduler == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.learning_rate,
                total_steps=training_steps_total
            )
        elif self.scheduler == "DynamicWarmupStableDecayScheduler":
            scheduler = DynamicWarmupStableDecayScheduler(
                optimizer=optimizer,
                lr=self.learning_rate,
                warmup_steps=300,
            )
        else:
            raise Exception("Unknown scheduler")
        
        return optimizer, scheduler

    def calculate_metrics(self, batch, masked_language_modeling_output, next_sentence_prediction_output):
        metrics = dict()

        masked_tokens = batch["masked_tokens_mask"]
        masked_token_predictions = masked_language_modeling_output
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


class BertModelForSequenceClassification(L.LightningModule):

    def __init__(
        self,
        pretrained_model: BertModel,
        num_classes: int,
        learning_rate: float = 1e-4,
        scheduler: str = "CosineAnnealingLR",
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.config = pretrained_model.config
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        assert scheduler in ["CosineAnnealingLR", "OneCycleLR", "DynamicWarmupStableDecayScheduler"]
        self.scheduler = scheduler

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

        # update dropout probabilities
        for module in self.bert.modules():
            if isinstance(module, nn.Dropout):
                module.p = p_dropout

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

        metrics = self.calculate_metrics(sequence_classification_output, batch["labels"], prefix="train")

        return classification_loss, metrics

    def validation_step(self, batch, batch_idx):
        sequence_classification_output = self(**batch)

        classification_loss = self.classification_loss_fn(
            sequence_classification_output, batch["labels"]
        )

        metrics = self.calculate_metrics(sequence_classification_output, batch["labels"], prefix="val")

        return classification_loss, metrics

    def configure_optimizers(self, training_steps_total):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12
        )

        if self.scheduler == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=training_steps_total
            )
        elif self.scheduler == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.learning_rate,
                total_steps=training_steps_total
            )
        elif self.scheduler == "DynamicWarmupStableDecayScheduler":
            scheduler = DynamicWarmupStableDecayScheduler(
                optimizer=optimizer,
                lr=self.learning_rate,
                warmup_steps=300,
            )
        else:
            raise Exception("Unknown scheduler")

        return optimizer, scheduler

    def calculate_metrics(self, sequence_classification_output, labels, prefix="train"):
        accuracy = self.classification_accuracy(
            sequence_classification_output, labels
        )
        metrics = {f"{prefix}/accuracy": accuracy}
        return metrics
