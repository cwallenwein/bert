import time

import torch
import torchmetrics
from torch import nn

from bert.model.bert.config import BertConfig
from bert.model.bert.for_mlm_pretraining import BertModelForMLM


class BertModelForMLMandandNSP(BertModelForMLM):

    def __init__(
        self,
        config: BertConfig,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        scheduler: str = "CosineAnnealingLR",
        compile: bool = False,
    ):
        super().__init__(config, batch_size, learning_rate, scheduler, compile=compile)

        # next sentence prediction
        self.next_sentence_prediction_head = nn.Linear(config.d_model, 1, bias=False)
        self.nsp_loss_fn = nn.BCEWithLogitsLoss()
        self.nsp_accuracy = torchmetrics.Accuracy(task="binary", average="micro")

        # only compile submodules to fix lightning errors when calling self.log
        if compile:
            self.next_sentence_prediction_head = torch.compile(
                self.next_sentence_prediction_head
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        masked_tokens_mask,
        token_type_ids=None,
        **kwargs
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        mlm_output = self.masked_language_modeling_head(bert_output[masked_tokens_mask])
        nsp_output = self.next_sentence_prediction_head(bert_output[..., 0, :]).squeeze(
            -1
        )
        return mlm_output, nsp_output

    def training_step(self, batch, batch_idx):
        step_start_time = time.time()
        masked_language_modeling_output, next_sentence_prediction_output = self(**batch)

        masked_tokens = batch["masked_tokens_mask"]
        masked_token_labels = batch["labels"][masked_tokens]
        mlm_loss = self.mlm_loss_fn(
            masked_language_modeling_output, masked_token_labels
        )
        step_duration = time.time() - step_start_time
        mlm_acc = self.mlm_accuracy(
            masked_language_modeling_output, masked_token_labels
        )

        next_sentence_prediction_labels = batch["labels"][..., 0]
        nsp_loss = self.nsp_loss_fn(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )
        nsp_acc = self.nsp_accuracy(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )

        total_loss = mlm_loss + nsp_loss

        self.log("train/loss", total_loss)
        self.log("train/mlm_acc", mlm_acc)
        self.log("train/nsp_acc", nsp_acc)
        self.log("train/step_duration", step_duration)
        self.log("train/step_duration_per_sample", step_duration / self.batch_size)

        return total_loss
