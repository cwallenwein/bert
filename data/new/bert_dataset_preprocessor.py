from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import random
from data.new.dataset_tokenizer import DatasetTokenizer
from data.new.mlm_preprocessor import MaskedLanguageModelingPreprocessor


class BertDatasetPreprocessor:
    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizerFast, context_length: int):
        assert context_length & 8 == 0
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_length = context_length

    def preprocess(self, num_tokens: int):
        assert num_tokens & self.context_length == 0

        # tokenize dataset
        dataset_tokenizer = DatasetTokenizer(self.dataset, tokenizer=self.tokenizer)
        tokenized_dataset = dataset_tokenizer(context_length=self.context_length, batch_size=4000, num_proc=8)

        # upsample dataset
        random_indices = self.get_random_indices(dataset=tokenized_dataset, num_tokens=num_tokens)
        tokenized_dataset = tokenized_dataset.select(random_indices)

        # mask dataset
        mlm_preprocessor = MaskedLanguageModelingPreprocessor(tokenized_dataset, self.tokenizer)
        masked_dataset = mlm_preprocessor(batch_size=1000, num_proc=8)

        return masked_dataset

    def get_random_indices(self, dataset: Dataset, num_tokens: int):
        len_dataset = len(dataset)
        num_samples = num_tokens // self.context_length

        if num_samples < len_dataset:
            # sample without replacement
            random_indices = random.sample(range(len_dataset), k=num_samples)
        else:
            # sample with replacement
            random_indices = random.choices(range(len_dataset), k=num_samples)

        return random_indices

