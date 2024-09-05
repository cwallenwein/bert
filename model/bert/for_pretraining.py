import torch
from torch import nn, optim
from model.bert.config import BertConfig
from model.bert.model import BertModel
import lightning.pytorch as pl
import torchmetrics
from trainer.scheduler import DynamicWarmupStableDecayScheduler

# Typehints
from jaxtyping import Float
from torch import Tensor


class BertModelForMLM(pl.LightningModule):
    # TODO: implement and test flash attention
    # TODO: add an option for weight tying to the config
    def __init__(
        self,
        config: BertConfig,
        learning_rate: float = 1e-4,
        scheduler: str = "CosineAnnealingLR"
    ):
        super().__init__()
        self.config: BertConfig = config
        self.bert = BertModel(config)
        self.learning_rate = learning_rate

        assert scheduler in ["CosineAnnealingLR", "OneCycleLR", "DynamicWarmupStableDecayScheduler"]
        self.scheduler = scheduler

        # masked language modeling
        self.masked_language_modeling_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # TODO: check if softmax and cross entropy loss is combined?
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        self.mlm_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=config.vocab_size,
            average="micro"
        )

        self.save_hyperparameters()

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
        mlm_output = self.masked_language_modeling_head(
            bert_output[masked_tokens_mask]
        )

        return mlm_output

    def training_step(self, batch, batch_idx):
        masked_language_modeling_output = self(**batch)

        # metrics = self.calculate_metrics(batch, masked_language_modeling_output, next_sentence_prediction_output)
        masked_tokens = batch["masked_tokens_mask"]
        
        masked_token_labels = batch["labels"][masked_tokens]
        mlm_loss = self.mlm_loss_fn(masked_language_modeling_output, masked_token_labels)

        # calculate accuracy
        mlm_acc = self.mlm_accuracy(
            masked_language_modeling_output, masked_token_labels
        )

        self.log("train/loss", mlm_loss)
        self.log("train/mlm_acc", mlm_acc)

        return mlm_loss
        
    def configure_optimizers(self, training_steps_total = None):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

class BertModelForMLMandandNSP(BertModelForMLM):
    
    def __init__(self, config: BertConfig, learning_rate: float = 1e-4, scheduler: str = "CosineAnnealingLR"):
        super().__init__(config, learning_rate, scheduler)

        # next sentence prediction
        self.next_sentence_prediction_head = nn.Linear(config.d_model, 1, bias=False)
        self.nsp_loss_fn = nn.BCEWithLogitsLoss()
        self.nsp_accuracy = torchmetrics.Accuracy(task="binary", average="micro")

    def forward(self, input_ids, attention_mask, masked_tokens_mask, token_type_ids=None, **kwargs):
        bert_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        mlm_output = self.masked_language_modeling_head(
            bert_output[masked_tokens_mask]
        )
        nsp_output = self.next_sentence_prediction_head(
            bert_output[..., 0, :]
        ).squeeze(-1)
        return mlm_output, nsp_output

    def training_step(self, batch, batch_idx):
        masked_language_modeling_output, next_sentence_prediction_output = self(**batch)

        masked_tokens = batch["masked_tokens_mask"]
        masked_token_labels = batch["labels"][masked_tokens]
        mlm_loss = self.mlm_loss_fn(masked_language_modeling_output, masked_token_labels)
        mlm_acc = self.mlm_accuracy(masked_language_modeling_output, masked_token_labels)

        next_sentence_prediction_labels = batch["labels"][..., 0]
        nsp_loss = self.nsp_loss_fn(next_sentence_prediction_output, next_sentence_prediction_labels)
        nsp_acc = self.nsp_accuracy(next_sentence_prediction_output, next_sentence_prediction_labels)

        total_loss = mlm_loss + nsp_loss

        self.log("train/loss", total_loss)
        self.log("train/mlm_acc", mlm_acc)
        self.log("train/nsp_acc", nsp_acc)

        return total_loss
