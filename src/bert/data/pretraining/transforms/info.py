from bert.data.pretraining.masked_language_modeling.config import (
    MaskedLanguageModelingDatasetConfig,
)
from datasets import Dataset


class AddConfigTransform:
    def __init__(self, config: MaskedLanguageModelingDatasetConfig):
        self.config = config

    def __call__(self, dataset: Dataset):
        dataset.info.description = self.config.to_string()
        return dataset
