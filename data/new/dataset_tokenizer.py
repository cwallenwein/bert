from datasets import Dataset
from data.util import is_power_of_2
from transformers import PreTrainedTokenizerFast
from typing import Optional
from data.tokenizer_util import tokenize


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

    def __call__(self, context_length: int = 128, packing: bool = False, batch_size: int = 1_000, num_proc: Optional[int] = None) -> Dataset:
        """ pack multiple samples into one """
        assert is_power_of_2(context_length), f"context_length({context_length}) must be a power of 2"
        # TODO implement packing
        assert not packing, "Packing is not implemented yet"

        dataset = self.dataset.map(
            self.tokenize_fn,
            input_columns="text",
            batched=True,
            batch_size=batch_size,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset",
            fn_kwargs={"context_length": context_length},
            num_proc=num_proc,
        ).with_format("torch")

        assert sorted(dataset.column_names) == [
            "attention_mask", "input_ids", "special_tokens_mask", "token_type_ids"
        ], f"Actual column names: {dataset.column_names}"
        return dataset

    def tokenize_fn(self, batch, context_length, return_tensors = None):
        return tokenize(batch, tokenizer=self.tokenizer, context_length=context_length, return_tensors=return_tensors)
