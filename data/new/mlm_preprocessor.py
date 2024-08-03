from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from typing import Optional
import torch


class MaskedLanguageModelingPreprocessor:
    """
    Takes a tokenized dataset and applies masked language modeling
    Input columns:
        - input_ids
        - token_type_ids
        - attention_mask
        - special_tokens_mask

    Output columns:
        - labels
        - masked_tokens_mask

        - input_ids
        - token_type_ids
        - attention_mask
        - special_tokens_mask
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerFast,
        p_mask: float = 0.15,
        p_replacement_mask: float = 0.8,
        p_replacement_random: float = 0.1,
        p_replacement_unchanged: float = 0.1,
        cleanup_cache: bool = False
    ):
        assert sorted(dataset.column_names) == [
            "attention_mask", "input_ids", "special_tokens_mask", "token_type_ids"
        ], f"Actual column names: {dataset.column_names}"
        assert 0 <= p_mask <= 1
        assert 0 <= p_replacement_mask <= 1
        assert 0 <= p_replacement_random <= 1
        assert 0 <= p_replacement_unchanged <= 1
        assert p_replacement_mask + p_replacement_random + p_replacement_unchanged == 1

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.p_mask = p_mask
        self.p_replacement_mask = p_replacement_mask
        self.p_replacement_random = p_replacement_random
        self.p_replacement_unchanged = p_replacement_unchanged

        if cleanup_cache:
            self.dataset.cleanup_cache_files()

    def __call__(self, batch_size: int = 1_000, num_proc: Optional[int] = None) -> Dataset:

        dataset = self.dataset.map(
            self.mask_fn,
            batched=True,
            batch_size=batch_size,
            desc="Masking dataset",
            num_proc=num_proc,
        )

        assert sorted(dataset.column_names) == [
            "attention_mask", "input_ids", "labels", "masked_tokens_mask", "special_tokens_mask", "token_type_ids"
        ], f"Actual column names: {dataset.column_names}"
        return dataset

    def mask_fn(self, batch: dict):
        # returns input_ids, labels, masked_tokens_mask
        batch_update = dict()
        batch_update["labels"] = batch["input_ids"].clone()

        # create mask
        # mask == 0 -> don't mask
        # mask == 1 -> [MASK]
        # mask == 2 -> random replace
        # mask == 3 -> unchanged

        p_mask_replace = self.p_mask * self.p_replacement_mask
        p_random_replace = self.p_mask * self.p_replacement_random
        p_unchanged = self.p_mask * self.p_replacement_unchanged

        random_values = torch.rand(size=batch["input_ids"].size())
        mask_replace = random_values < p_mask_replace
        random_replace = (p_mask_replace < random_values) & (random_values < p_mask_replace + p_random_replace)
        unchanged = (p_mask_replace + p_random_replace < random_values) & (
            random_values < p_mask_replace + p_random_replace + p_unchanged)
        special_tokens = batch["special_tokens_mask"] == 1

        mask = torch.where(mask_replace, 1, 0)
        mask = torch.where(random_replace, 2, mask)
        mask = torch.where(unchanged, 3, mask)
        mask = torch.where(special_tokens, 0, mask)

        batch_update["masked_tokens_mask"] = ((mask == 1) | (mask == 2) | (mask == 3)).int()

        # replace values according to mask
        input_ids = batch["input_ids"]

        input_ids[mask == 1] = self.tokenizer.mask_token_id
        random_token = torch.randint_like(input_ids, len(self.tokenizer))
        input_ids[mask == 2] = random_token[mask == 2]

        batch_update["input_ids"] = input_ids

        # define mask (which tokens should be predicted)
        # batch["attention_mask"] = mask != 0
        return batch_update
