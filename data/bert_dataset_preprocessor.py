from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import random
from data.dataset_tokenizer import DatasetTokenizer
from data.mlm_preprocessor import MaskedLanguageModelingPreprocessor


class BertDatasetPreprocessor:
    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizerFast, context_length: int):
        assert context_length & 8 == 0
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_length = context_length

    def preprocess(self, num_tokens: int):
        assert num_tokens % self.context_length == 0
        num_samples = num_tokens // self.context_length

        dataset = self.dataset.shuffle(seed=42)

        # tokenize dataset
        dataset_tokenizer = DatasetTokenizer(dataset, tokenizer=self.tokenizer)
        tokenized_dataset = dataset_tokenizer(context_length=self.context_length, batch_size=128, num_proc=8)

        if len(tokenized_dataset) < num_samples:
            # upsample dataset
            random_indices = self.get_random_indices(dataset=tokenized_dataset, num_samples=num_samples)
            tokenized_dataset = tokenized_dataset.select(random_indices)

        # mask dataset
        mlm_preprocessor = MaskedLanguageModelingPreprocessor(tokenized_dataset, self.tokenizer)
        masked_dataset = mlm_preprocessor(num_samples=num_samples, context_length=self.context_length, batch_size=128, num_proc=8)

        return masked_dataset

    @staticmethod
    def get_random_indices(dataset: Dataset, num_samples: int):
        len_dataset = len(dataset)

        if num_samples < len_dataset:
            # sample without replacement
            random_indices = random.sample(range(len_dataset), k=num_samples)
        else:
            # sample with replacement
            random_indices = random.choices(range(len_dataset), k=num_samples)

        return random_indices

