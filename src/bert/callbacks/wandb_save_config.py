import os
import warnings

from lightning import pytorch as lightning
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback


class WandbSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:

        saved_config = False
        for logger in trainer.loggers:
            if isinstance(logger, lightning.loggers.WandbLogger):
                wandb_logger = logger

                # save config locally
                save_dir = wandb_logger.save_dir
                name = wandb_logger.name
                experiment_id = wandb_logger.experiment.id
                config_dir = f"{save_dir}/{name}/{experiment_id}"

                os.makedirs(config_dir, exist_ok=True)

                config_path = f"{config_dir}/config.yaml"
                self.parser.save(self.config, config_path, overwrite=True)

                # log config as wandb artifact
                wandb_logger.experiment.log_artifact(config_path).wait

                saved_config = True

        if not saved_config:
            warnings.warn(
                "Unable to save config. Most likely because WandbLogger was not used."
            )
