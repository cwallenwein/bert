import time
from typing import Annotated

import lightning.pytorch as pl
import torch
import torchmetrics
from progressive_scheduling import CosineAnnealingLR, OneCycleLR
from torch import Tensor, nn, optim

from model.bert.config import BertConfig
from model.bert.model import BertModel


class BertModelForMLM(pl.LightningModule):
    # TODO: implement and test flash attention
    # TODO: add an option for weight tying to the config
    def __init__(
        self,
        config: BertConfig,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        scheduler: str = "CosineAnnealingLR",
        compile: bool = False,
    ):
        super().__init__()
        self.config: BertConfig = config
        self.bert = BertModel(config)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        assert scheduler in ["CosineAnnealingLR", "OneCycleLR"]
        # TODO: fix support for DynamicWarmupStableDecayScheduler
        self.scheduler = scheduler

        # masked language modeling
        self.masked_language_modeling_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        # TODO: check if softmax and cross entropy loss is combined?
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.mlm_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.vocab_size, average="micro"
        )

        # only compile submodules to fix lightning errors when calling self.log
        if compile:
            self.bert = torch.compile(self.bert)
            self.masked_language_modeling_head = torch.compile(
                self.masked_language_modeling_head
            )

        self.save_hyperparameters()

    def forward(
        self,
        input_ids: Annotated[Tensor, "batch sequence_length"],
        attention_mask: Annotated[Tensor, "batch sequence_length"],
        masked_tokens_mask: Annotated[Tensor, "batch sequence_length"],
        token_type_ids: Annotated[Tensor, "batch sequence_length"] = None,
        **kwargs
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        mlm_output = self.masked_language_modeling_head(bert_output[masked_tokens_mask])

        return mlm_output

    def training_step(self, batch, batch_idx):
        step_start_time = time.time()
        masked_language_modeling_output = self(**batch)

        masked_tokens = batch["masked_tokens_mask"]

        masked_token_labels = batch["labels"][masked_tokens]
        mlm_loss = self.mlm_loss_fn(
            masked_language_modeling_output, masked_token_labels
        )
        step_duration = time.time() - step_start_time
        # calculate accuracy
        mlm_acc = self.mlm_accuracy(
            masked_language_modeling_output, masked_token_labels
        )

        self.log("train/loss", mlm_loss)
        self.log("train/mlm_acc", mlm_acc)
        self.log("train/step_duration", step_duration)
        self.log("train/step_duration_per_sample", step_duration / self.batch_size)

        return mlm_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-12,
            weight_decay=0.01,
        )

        if self.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer)
        elif self.scheduler == "OneCycleLR":
            scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate)
        else:
            raise Exception("Unknown scheduler")

        scheduler = {"scheduler": scheduler, "name": "train/lr", "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
