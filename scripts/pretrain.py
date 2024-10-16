from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader

from bert.callbacks import WandbSaveConfigCallback
from bert.model.bert import BertModelForMLM
from datasets import Dataset


class DataModule(LightningDataModule):

    def __init__(self, dataset_dir, batch_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = Dataset.load_from_disk(self.dataset_dir)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader


def cli_main():

    LightningCLI(
        BertModelForMLM,
        DataModule,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"save_to_log_dir": False},
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli_main()
