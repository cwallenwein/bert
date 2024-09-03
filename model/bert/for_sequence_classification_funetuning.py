import torch
import torchmetrics
from torch import nn, optim
from model.bert.model import BertModel
import lightning.pytorch as pl
from trainer.scheduler import DynamicWarmupStableDecayScheduler

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModelForSequenceClassification(pl.LightningModule):

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

        self.save_hyperparameters()

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

        accuracy = self.classification_accuracy(
            sequence_classification_output, batch["labels"]
        )
        self.log("train/loss", classification_loss)
        self.log("train/accuracy", accuracy)

        return classification_loss

    def validation_step(self, batch, batch_idx):
        sequence_classification_output = self(**batch)

        classification_loss = self.classification_loss_fn(
            sequence_classification_output, batch["labels"]
        )

        accuracy = self.classification_accuracy(
            sequence_classification_output, batch["labels"]
        )
        self.log("val/loss", classification_loss)
        self.log("val/accuracy", accuracy)

        return classification_loss

    def configure_optimizers(self, training_steps_total = None):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12,
            weight_decay=0.01
        )

        if training_steps_total is None and self.trainer is not None:
            training_steps_total = self.trainer.estimated_stepping_batches

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

        scheduler =  {"scheduler": scheduler, "name": "train/lr", "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}, 