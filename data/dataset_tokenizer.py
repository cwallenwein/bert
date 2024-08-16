from datasets import Dataset
from data.util import is_power_of_2, add_early_stopping
from transformers import PreTrainedTokenizerFast
from typing import Optional
from datasets import Features, Value, Sequence


class DatasetTokenizer:
    """
    Takes a text dataset and tokenizes it
    Input columns:
        - text

    Output columns:
        - input_ids
        - attention_mask
        - special_tokens_mask
    """

    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizerFast, cleanup_cache: bool = False):
        assert "text" in dataset.column_names, f"Actual column names: {dataset.column_names}"
        self.dataset = dataset
        self.tokenizer = tokenizer

        if cleanup_cache:
            self.dataset.cleanup_cache_files()

    def __call__(
        self,
        context_length: int = 128,
        packing: bool = False,
        batch_size: int = 1_000,
        num_proc: Optional[int] = None,
        save: bool = False
    ) -> Dataset:
        """ pack multiple samples into one """
        assert is_power_of_2(context_length), f"context_length({context_length}) must be a power of 2"
        # TODO implement packing
        assert not packing, "Packing is not implemented yet"

        dataset = self.dataset.map(
            self.tokenize,
            input_columns="text",
            batched=True,
            batch_size=batch_size,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=num_proc,
            keep_in_memory=True,
            fn_kwargs={
                "context_length": context_length
            },
            features=Features({
                "attention_mask": Sequence(feature=Value(dtype="bool", id=None), length=context_length, id=None),
                "input_ids": Sequence(feature=Value(dtype="int16", id=None), length=context_length, id=None),
                "special_tokens_mask": Sequence(feature=Value(dtype="bool", id=None), length=context_length, id=None)
            })
        ).with_format("torch")

        # if save:
        #     dataset.save_to_disk(f"./tokenized/{self.dataset.}/{self.tokenizer.name_or_path}/{context_length}")

        assert sorted(dataset.column_names) == [
            "attention_mask", "input_ids", "special_tokens_mask"
        ], f"Actual column names: {dataset.column_names}"
        return dataset

    def tokenize(self, batch, context_length: int, return_tensors=None):
        result = self.tokenizer(
            batch,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_length=context_length,
            padding="max_length",
            return_overflowing_tokens=True,
            truncation=True,
            return_tensors=return_tensors
        )

        result.pop("overflow_to_sample_mapping")
        return result
