import warnings
from typing import Annotated

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor, nn, optim

from bert.model.bert.config import BertConfig
from bert.model.bert.model import BertModel


class BertModelForMLM(LightningModule):
    # TODO: implement and test flash attention
    # TODO: add an option for weight tying to the config
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        context_length: int = 128,
        vocab_size: int = 30522,
        n_segments: int = 2,
        initializer_range: float = 0.02,
        feed_forward_activation: str = "gelu",
        feed_forward_intermediate_size: int = 512,
        feed_forward_bias: bool = True,
        p_embedding_dropout: float = 0.1,
        p_attention_dropout: float = 0.1,
        p_feed_forward_dropout: float = 0.1,
        attention_implementation: str = "pytorch",
        positional_information_type: str = "learned",
        attention_bias: bool = True,
        layer_norm: str = "pre",
        add_final_layer_norm: bool = False,
        compile: bool = False,
        optimizer: OptimizerCallable = optim.Adam,
        scheduler: LRSchedulerCallable = optim.lr_scheduler.ConstantLR,
    ):
        super().__init__()
        self.config: BertConfig = BertConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            context_length=context_length,
            vocab_size=vocab_size,
            n_segments=n_segments,
            initializer_range=initializer_range,
            feed_forward_activation=feed_forward_activation,
            feed_forward_intermediate_size=feed_forward_intermediate_size,
            feed_forward_bias=feed_forward_bias,
            p_embedding_dropout=p_embedding_dropout,
            p_attention_dropout=p_attention_dropout,
            p_feed_forward_dropout=p_feed_forward_dropout,
            attention_implementation=attention_implementation,
            positional_information_type=positional_information_type,
            attention_bias=attention_bias,
            layer_norm=layer_norm,
            add_final_layer_norm=add_final_layer_norm,
            with_next_sentence_prediction=False,
        )

        self.bert = BertModel(self.config)
        self.masked_language_modeling_head = nn.Linear(d_model, vocab_size, bias=False)
        # TODO: check if softmax and cross entropy loss is combined?
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.mlm_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=vocab_size, average="micro"
        )

        # only compile submodules to fix lightning errors when calling self.log
        if compile:
            if not hasattr(torch, "compile"):
                warnings.warn(
                    "Warning: torch.compile is not available. Skipping compilation."
                )

            self.bert = torch.compile(self.bert)
            self.masked_language_modeling_head = torch.compile(
                self.masked_language_modeling_head
            )

        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(
        self,
        input_ids: Annotated[Tensor, "batch sequence_length"],
        attention_mask: Annotated[Tensor, "batch sequence_length"],
        masked_tokens_mask: Annotated[Tensor, "batch sequence_length"],
        token_type_ids: Annotated[Tensor, "batch sequence_length"] = None,
        **kwargxs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        bert_output = bert_output[masked_tokens_mask]
        mlm_output = self.masked_language_modeling_head(bert_output)

        return mlm_output

    def training_step(self, batch, batch_idx):
        masked_language_modeling_output = self(**batch)

        masked_tokens = batch["masked_tokens_mask"]

        masked_token_labels = batch["labels"][masked_tokens]
        mlm_loss = self.mlm_loss_fn(
            masked_language_modeling_output, masked_token_labels
        )
        # calculate accuracy
        mlm_acc = self.mlm_accuracy(
            masked_language_modeling_output, masked_token_labels
        )

        self.log("train/loss", mlm_loss.item())
        self.log("train/mlm_acc", mlm_acc)

        return mlm_loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        lr_scheduler_config = {
            "scheduler": self.scheduler(optimizer),
            "interval": "step",
            "name": "train/lr",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def log(self, *args, **kwargs):
        if self._trainer is not None:
            super().log(*args, **kwargs)
        else:
            assert len(args) == 2
            import wandb

            metric_name = args[0]
            metric_value = args[1]
            wandb.log({metric_name: metric_value})

    def save_hyperparameters(self, *args, **kwargs):
        if self._trainer is not None:
            super().save_hyperparameters(*args, **kwargs)
        else:
            warnings.warn("save_hyperparameters is not implemented yet")
