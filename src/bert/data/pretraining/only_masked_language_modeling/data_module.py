from pathlib import Path

from fastcore.transform import Pipeline
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from bert.data.pretraining.load_raw import load_local_hf_dataset
from bert.data.pretraining.only_masked_language_modeling import (
    MaskedLanguageModelingDatasetConfig,
)
from bert.data.pretraining.transforms import (
    AddConfigTransform,
    MaskedLanguageModelingTransform,
    TokenizeTransform,
)
from bert.tokenizer.load import load_tokenizer
from bert.utils import get_datasets_processed_dir
from datasets import Dataset, load_from_disk

# data.context_length=128 data.num_samples=10000 data.p_mask=0.15 data.p_replacement_mask=0.8 data.p_replacement_random=0.1 data.raw_dataset_name=roleplay data.tokenizer_id_or_name=bert-base-uncased


class MaskedLanguageModelingDataset(LightningDataModule):
    """
    Module takes dataset name and tokenizer name and loads the respective things


    TODO: add support for next sentence prediction

    """

    processed_datasets_subfolder = "masked_l"

    def __init__(
        self,
        # for MaskedLanguageModelingDatasetConfig
        tokenizer_id_or_name: str,
        raw_dataset_name: str,
        num_samples: int = None,
        context_length: int = 128,
        p_mask: float = 0.15,
        p_replacement_mask: float = 0.8,
        p_replacement_random: float = 0.1,
        # transforms
        # sort_by: Optional[str] = None,
        # sort_order: Optional[str] = None,
        # filter_by: Optional[str] = None,
        # filter_value: Optional[float] = None,
        batch_size: int = 16,
    ):
        """
        dataset type used throughout this data module: datasets.Dataset
        """
        super().__init__()

        # self.tokenizer_id_or_name = tokenizer_id_or_name
        # self.raw_dataset_name = raw_dataset_name
        # self.context_length = context_length
        # self.p_mask = p_mask
        # self.p_replacement_mask = p_replacement_mask
        # self.p_replacement_random = p_replacement_random

        self.dataset_config = MaskedLanguageModelingDatasetConfig(
            tokenizer_id_or_name=tokenizer_id_or_name,
            raw_dataset_name=raw_dataset_name,
            num_samples=num_samples,
            context_length=context_length,
            p_mask=p_mask,
            p_replacement_mask=p_replacement_mask,
            p_replacement_random=p_replacement_random,
        )

        # transforms
        # assert sort_order in ["asc", "desc"]
        # TODO: make sure that sort_by is a valid column
        # self.sort_by = sort_by
        # self.sort_order = sort_order
        # self.filter_by = filter_by
        # self.filter_value = filter_value
        self.batch_size = batch_size

        self.tokenizer = load_tokenizer(self.dataset_config.tokenizer_id_or_name)

        self.save_hyperparameters()

    def prepare_data(self):
        """
        1. Load processed dataset if one with the same PretrainingDatasetConfig exists
        2. Otherwise preprocess according to PretrainingDatasetConfig
            - tokenize
            - mask tokens
        3. Save dataset to preprocessed_datasets_dir/dataset_name

        """
        try:
            self.load_dataset(self.dataset_config)
        except FileNotFoundError:
            dataset = load_local_hf_dataset(self.dataset_config.raw_dataset_name)

            tokenize_dataset = TokenizeTransform(
                tokenizer=self.tokenizer,
                num_samples=self.dataset_config.num_samples,
                context_length=self.dataset_config.context_length,
            )

            apply_masked_language_modeling = MaskedLanguageModelingTransform(
                tokenizer=self.tokenizer,
                context_length=self.dataset_config.context_length,
                p_mask=self.dataset_config.p_mask,
                p_replacement_mask=self.dataset_config.p_replacement_mask,
                p_replacement_random=self.dataset_config.p_replacement_random,
            )

            add_dataset_config = AddConfigTransform(self.dataset_config)

            pipeline = Pipeline(
                [
                    tokenize_dataset,
                    apply_masked_language_modeling,
                    add_dataset_config,
                ]
            )
            dataset = pipeline(dataset)
            self.save_dataset(dataset=dataset, dataset_config=self.dataset_config)

    def setup(self, stage):
        """
        Loads the dataset from the filesystem
        Applies transformations (sorting, filtering)
        """
        train_dataset = MaskedLanguageModelingDataset.load_dataset(self.dataset_config)

        # apply transformations here

        self.train_dataset = train_dataset

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        return dataloader

    def teardown(self):
        pass

    @staticmethod
    def load_dataset(dataset_config: MaskedLanguageModelingDatasetConfig):
        dataset_path = MaskedLanguageModelingDataset.get_dataset_path(dataset_config)
        if not dataset_path.exists():
            raise FileNotFoundError("No dataset with the given config exists.")
        dataset = load_from_disk(dataset_path=dataset_path)
        return dataset

    @staticmethod
    def save_dataset(
        dataset: Dataset, dataset_config: MaskedLanguageModelingDatasetConfig
    ):
        dataset_path = MaskedLanguageModelingDataset.get_dataset_path(dataset_config)
        dataset.save_to_disk(dataset_path=dataset_path)

    @staticmethod
    def get_dataset_path(dataset_config: MaskedLanguageModelingDatasetConfig) -> Path:
        dataset_config_hash = dataset_config.get_hash_value()
        datasets_processed_dir = get_datasets_processed_dir()
        dataset_path = datasets_processed_dir / __class__.__name__ / dataset_config_hash
        return dataset_path
