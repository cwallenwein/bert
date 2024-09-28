import argparse

import lightning.pytorch as pl
import torch
from datasets import Dataset
from progressive_scheduling.callbacks.lightning import AutoSchedulingCallback
from torch.utils.data import DataLoader

from model.bert import BertConfig, BertModelForMLM

cramming_config = BertConfig(
    d_model=768,
    n_layers=12,
    context_length=128,
    n_heads=12,
    feed_forward_intermediate_size=3072,
    attention_implementation="pytorch",
    positional_information_type="scaled_sinusoidal",
    p_embedding_dropout=0.0,
    p_attention_dropout=0.0,
    p_feed_forward_dropout=0.0,
    attention_bias=False,
    feed_forward_bias=False,
    feed_forward_activation="glu",
    add_final_layer_norm=True,
)

small_config = BertConfig(
    d_model=2,
    n_layers=1,
    context_length=128,
    n_heads=1,
    feed_forward_intermediate_size=2,
    attention_implementation="pytorch",
    positional_information_type="scaled_sinusoidal",
    p_embedding_dropout=0.0,
    p_attention_dropout=0.0,
    p_feed_forward_dropout=0.0,
    attention_bias=False,
    feed_forward_bias=False,
    # feed_forward_activation="glu",
    add_final_layer_norm=True,
)

working_config = BertConfig(
    d_model=768,
    n_layers=12,
    context_length=128,
    n_heads=12,
    feed_forward_intermediate_size=3072,
    attention_implementation="pytorch",
    positional_information_type="learned",
    p_embedding_dropout=0.0,
    p_attention_dropout=0.0,
    p_feed_forward_dropout=0.0,
    attention_bias=False,
    feed_forward_bias=False,
    feed_forward_activation="gelu",
    add_final_layer_norm=True,
)


# TODO: log time per training sample
def pretrain(
    max_steps: int = -1,
    max_time_in_min: int = None,
    micro_batch_size: int = 352,
    macro_batch_size: int = 3_872,
    learning_rate: float = 0.00087,
    scheduler: str = "CosineAnnealingLR",
    limit_train_batches: float = 1.0,
):
    if max_steps == -1:
        assert max_time_in_min is not None
    else:
        assert max_time_in_min is None

    assert macro_batch_size % micro_batch_size == 0
    gradient_accumulation_steps = macro_batch_size // micro_batch_size

    model = BertModelForMLM(
        small_config, scheduler=scheduler, learning_rate=learning_rate
    )

    # prepare model
    if not torch.backends.mps.is_available():
        model = torch.compile(model)

    # load data
    dataset = Dataset.load_from_disk("./data/datasets/mlm-fineweb-10BT")
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, num_workers=5)
    print("len(dataloader)", len(dataloader))

    # setup logging
    wandb_logger = pl.loggers.WandbLogger(
        project="BERT",
        log_model=True,
        job_type="pretraining",
        dir="..",
        config={"model": model.config.__dict__},
    )
    wandb_logger.watch(model, log="gradients")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # setup trainer

    auto_scheduler = AutoSchedulingCallback(
        max_steps=max_steps, max_time_in_min=max_time_in_min
    )

    if max_time_in_min is not None:
        max_time_in_min = {"minutes": max_time_in_min}

    trainer = pl.Trainer(
        max_steps=max_steps,
        max_time=max_time_in_min,
        accumulate_grad_batches=gradient_accumulation_steps,
        gradient_clip_val=0.5,
        logger=wandb_logger,
        precision="bf16-mixed",
        limit_train_batches=limit_train_batches,
        callbacks=[lr_monitor, auto_scheduler],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    wandb_logger.experiment.unwatch(model)


def main():
    # Define parser
    parser = argparse.ArgumentParser(description="Finetune BERT model")

    # Finetuning arguments
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_time_in_min", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=352)
    parser.add_argument("--macro_batch_size", type=int, default=3_872)
    parser.add_argument("--learning_rate", type=float, default=0.00087)
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)

    args = parser.parse_args()
    pretrain(**args.__dict__)


if __name__ == "__main__":
    main()
