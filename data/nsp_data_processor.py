import random
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, concatenate_datasets
from data.not_next_data_processor import NotNextDataProcessor
from data.is_next_data_processor import IsNextDataProcessor


class NextSentencePredictionDataProcessor:

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, dataset: Dataset, num_samples: int, context_length: int = 128):
        num_samples_per_type = num_samples // 2
        k_is_next = num_samples_per_type
        not_next_failure_rate = 0.35
        k_not_next = int(num_samples_per_type // not_next_failure_rate)

        k = k_is_next + k_not_next

        if k < len(dataset):
            # sample without replacement
            random_indices = random.sample(range(len(dataset)), k=k)
        else:
            # sample with replacement
            random_indices = random.choices(range(len(dataset)), k=k)

        split_next = IsNextDataProcessor(
            tokenizer=self.tokenizer, dataset=dataset.select(random_indices[:k_is_next])
        )
        split_not_next = NotNextDataProcessor(
            tokenizer=self.tokenizer, dataset=dataset.select(random_indices[k_is_next:])
        )

        is_next_dataset = split_next(context_length=context_length)
        not_next_dataset = split_not_next(context_length=context_length).select(range(num_samples_per_type))

        if split_not_next.num_batches > 0:
            print(split_not_next.num_not_possible / split_not_next.num_batches)

        dataset = concatenate_datasets([is_next_dataset, not_next_dataset]).shuffle()
        assert len(dataset) == num_samples
        return dataset
