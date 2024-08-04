from datasets import Dataset
from data.util import is_power_of_2, add_early_stopping
from transformers import PreTrainedTokenizerFast
from typing import Optional
from data.tokenizer_util import tokenize
from functools import partial


class DatasetTokenizer:
    """
    Takes a text dataset and tokenizes it
    Input columns:
        - text

    Output columns:
        - input_ids
        - token_type_ids
        - attention_mask
        - special_tokens_mask
    """

    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizerFast, cleanup_cache: bool = False):
        assert "text" in dataset.column_names, f"Actual column names: {dataset.column_names}"
        self.dataset = dataset
        self.tokenizer = tokenizer

        if cleanup_cache:
            self.dataset.cleanup_cache_files()

    def __call__(self, num_samples: int, context_length: int = 128, packing: bool = False, batch_size: int = 1_000, num_proc: Optional[int] = None) -> Dataset:
        """ pack multiple samples into one """
        assert is_power_of_2(context_length), f"context_length({context_length}) must be a power of 2"
        # TODO implement packing
        assert not packing, "Packing is not implemented yet"

        # define tokenization function
        self.tokenize_fn = partial(
            tokenize,
            tokenizer=self.tokenizer,
            context_length=context_length
        )

        assert num_samples & num_proc == 0
        max_samples_per_process = num_samples // num_proc
        sample_count = [0] * num_proc
        tokenize_with_early_stopping = partial(
            add_early_stopping,
            map_function=self.tokenize_fn,
            max_samples=max_samples_per_process,
            sample_count=sample_count,
            empty_sample={key: [] for key in ["attention_mask", "input_ids", "special_tokens_mask", "token_type_ids"]}
        )

        dataset = self.dataset.map(
            tokenize_with_early_stopping,
            input_columns="text",
            batched=True,
            batch_size=batch_size,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=num_proc,
            with_rank=True
        ).with_format("torch")

        assert sorted(dataset.column_names) == [
            "attention_mask", "input_ids", "special_tokens_mask", "token_type_ids"
        ], f"Actual column names: {dataset.column_names}"
        return dataset
