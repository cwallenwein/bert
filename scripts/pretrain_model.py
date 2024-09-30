# python scripts/pretrain_model.py --max_time_in_min=1 --micro_batch_size=320 --macro_batch_size=3840 --dataset_dir="./data/datasets/mlm-fineweb-10BT"

# TODO: track MFU
# TODO: how to properly track time?
# TODO: batch size schedule / gradient accumulation scheduler

import argparse

import lightning.pytorch as pl
from datasets import Dataset
from lightning.pytorch.callbacks import Timer
from progressive_scheduling.callbacks.lightning import AutoSchedulingCallback
from torch.utils.data import DataLoader

from model.bert.config import BertConfig
from model.bert.for_mlm_pretraining import BertModelForMLM

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
    dataset_dir: str,
    max_steps: int = -1,
    max_time_in_sec: int = None,
    micro_batch_size: int = 352,
    macro_batch_size: int = 3_872,
    learning_rate: float = 0.00087,
    scheduler: str = "CosineAnnealingLR",
    limit_train_batches: float = 1.0,
    compile: bool = False,
):
    if max_steps == -1:
        assert max_time_in_sec is not None
        time_based_training = True
    else:
        assert max_time_in_sec is None
        time_based_training = False

    assert macro_batch_size % micro_batch_size == 0
    gradient_accumulation_steps = macro_batch_size // micro_batch_size

    # prepare model
    # compile = not torch.backends.mps.is_available()
    model = BertModelForMLM(
        small_config,
        scheduler=scheduler,
        with_progressive_scheduling=time_based_training,
        learning_rate=learning_rate,
        compile=compile,
    )

    # load data
    dataset = Dataset.load_from_disk(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size)

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

    if time_based_training:
        timer = Timer({"seconds": max_time_in_sec})
        # automatically schedules the learning rate
        auto_scheduler = AutoSchedulingCallback(
            training_duration={"seconds": max_time_in_sec}
        )
        callbacks = [lr_monitor, auto_scheduler, timer]
    else:
        callbacks = [lr_monitor]

    # setup trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        accumulate_grad_batches=gradient_accumulation_steps,
        gradient_clip_val=0.5,
        logger=wandb_logger,
        precision="bf16-mixed",
        limit_train_batches=limit_train_batches,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    wandb_logger.experiment.unwatch(model)


def main():
    # Define parser
    parser = argparse.ArgumentParser(description="Finetune BERT model")

    # Finetuning arguments
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_time_in_sec", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=352)
    parser.add_argument("--macro_batch_size", type=int, default=3_872)
    parser.add_argument("--learning_rate", type=float, default=0.00087)
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--compile", default=False, type=lambda x: (str(x).lower() == "true")
    )

    args = parser.parse_args()
    pretrain(**args.__dict__)


if __name__ == "__main__":
    main()
