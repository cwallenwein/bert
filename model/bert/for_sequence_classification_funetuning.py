from typing import Annotated

import lightning.pytorch as pl
import torch
import torchmetrics
from torch import Tensor, nn, optim

from model.bert.model import BertModel
from model.lr_scheduler import DynamicWarmupStableDecayScheduler


class BertModelForSequenceClassification(pl.LightningModule):

    def __init__(
        self,
        pretrained_model: BertModel,
        num_classes: int,
        learning_rate: float = 1e-4,
        scheduler: str = "CosineAnnealingLR",
        p_dropout: float = 0.1,
        compile: bool = False,
    ):
        super().__init__()
        self.config = pretrained_model.config
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        assert scheduler in ["CosineAnnealingLR", "OneCycleLR"]
        # TODO: Fix support for DynamicWarmupStableDecayScheduler
        self.scheduler = scheduler

        self.bert = pretrained_model

        self.classification_head = nn.Linear(
            self.config.d_model, num_classes, bias=False
        )
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.classification_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )

        # freeze bert
        for param in self.bert.parameters():
            param.requires_grad = False

        # update dropout probabilities
        for module in self.bert.modules():
            if isinstance(module, nn.Dropout):
                module.p = p_dropout

        # only compile submodules to fix lightning errors when calling self.log
        if compile:
            self.bert = torch.compile(self.bert)
            self.classification_head = torch.compile(self.classification_head)

        self.save_hyperparameters()

    def forward(
        self,
        input_ids: Annotated[Tensor, "batch sequence_length"],
        attention_mask: Annotated[Tensor, "batch sequence_length"],
        **kwargs
    ):
        bert_output = self.bert(
            input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask
        )
        classification_output = self.classification_head(bert_output[..., 0, :])
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

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-12,
            weight_decay=0.01,
        )

        print("total_steps", self.trainer.estimated_stepping_batches)

        if self.scheduler == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.trainer.estimated_stepping_batches
            )
        elif self.scheduler == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler == "DynamicWarmupStableDecayScheduler":
            scheduler = DynamicWarmupStableDecayScheduler(
                optimizer=optimizer,
                lr=self.learning_rate,
                warmup_steps=300,
            )
        else:
            raise Exception("Unknown scheduler")

        scheduler = {"scheduler": scheduler, "name": "train/lr", "interval": "step"}
        return ({"optimizer": optimizer, "lr_scheduler": scheduler},)
