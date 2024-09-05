import argparse

import torch
from model.bert import BertConfig, BertModelForPretraining, BertModelForSequenceClassification, BertModel
from data import MNLIDataModule
import lightning.pytorch as pl


def finetune(
    wandb_run_name: str,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 5,
    scheduler: str = "CosineAnnealingLR",
    p_dropout: float = 0.1,
    limit_train_batches: float = 1.0,
    limit_val_batches: float = 1.0
):
    pretrained_bert = load_pretrained_model(wandb_run_name)
    model = BertModelForSequenceClassification(
        pretrained_model=pretrained_bert,
        num_classes=3,
        learning_rate=learning_rate,
        scheduler=scheduler,
        p_dropout=p_dropout,
    )

    # prepare model
    if not torch.backends.mps.is_available():
        model = torch.compile(model)

    # load data
    mnli_datamodule = MNLIDataModule(batch_size=batch_size)

    # setup logging
    wandb_logger = pl.loggers.WandbLogger(
        project="BERT",
        log_model=True,
        job_type="finetuning",
        dir="..",
        config={
                "model": model.config.__dict__
        },
        tags=["mnli"]
    )
    wandb_logger.watch(model, log="gradients")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # setup trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        precision = "bf16-mixed",
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        callbacks=[lr_monitor]
    )
    trainer.fit(model, datamodule=mnli_datamodule)
    wandb_logger.experiment.unwatch(model)
    

def load_pretrained_model(wandb_run_name: str) -> BertModel:
    # TODO: use pl_module.load_from_checkpoint to automatically get the correct config

    if torch.cuda.is_available():
        checkpoint = torch.load(f"experiments/{wandb_run_name}/checkpoint.pt", weights_only=True)
    else:
        checkpoint = torch.load(
            f"experiments/{wandb_run_name}/checkpoint.pt", map_location=torch.device("cpu")#, weights_only=True
        )

    default_config = BertConfig(
        d_model=768,
        n_layers=12,
        context_length=128,
        n_heads=12,
        feed_forward_intermediate_size=3072,
        attention_implementation="pytorch"
    )

    if "config" in checkpoint.keys():
        config = checkpoint["config"]
    else:
        config = default_config

    model = BertModelForPretraining(config)

    model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(model_state_dict)
    return model.bert

def main():
    # Define parser
    parser = argparse.ArgumentParser(description="Finetune BERT model")

    # Finetuning arguments
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--p_dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)

    args = parser.parse_args()
    finetune(**args.__dict__)


if __name__ == "__main__":
    main()
