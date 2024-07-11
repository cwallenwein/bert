import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import math


class MaskedLanguageModelingDataProcessor:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        p_mask: float = 0.15,
        p_replacement_mask: float = 0.8,
        p_replacement_random: float = 0.1,
        p_replacement_unchanged: float = 0.1,
    ):
        self.tokenizer = tokenizer
        self.p_mask = p_mask
        self.p_replacement_mask = p_replacement_mask
        self.p_replacement_random = p_replacement_random
        self.p_replacement_unchanged = p_replacement_unchanged

    def __call__(
        self,
        dataset: Dataset,
        context_length: int = 128,
        batch_size: int = 4,
    ):
        assert sorted(dataset.column_names) == [
            "attention_mask", "input_ids", "labels", "special_tokens", "token_type_ids"
        ]

        assert self.p_replacement_mask + self.p_replacement_random + self.p_replacement_unchanged == 1

        def is_power_of_2(number: int):
            exponent = math.log2(number)
            return int(exponent) == exponent
        assert is_power_of_2(context_length), "context_length must be a power of 2"

        size_before = len(dataset)
        dataset = dataset.map(
            self.mask_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=6,
            desc=f"Masking dataset",
        )
        size_after = len(dataset)
        assert size_before == size_after

        return dataset

    def mask_batch(
        self,
        batch: dict,
    ):
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
        special_tokens = batch["special_tokens"] == 1

        mask = torch.where(mask_replace, 1, 0)
        mask = torch.where(random_replace, 2, mask)
        mask = torch.where(unchanged, 3, mask)
        mask = torch.where(special_tokens, 0, mask)

        batch["masked_tokens"] = ((mask == 1) | (mask == 2) | (mask == 3)).int()

        # replace values according to mask
        input_ids = batch["input_ids"]

        input_ids[mask == 1] = self.tokenizer.mask_token_id
        random_token = torch.randint_like(input_ids, len(self.tokenizer))
        input_ids[mask == 2] = random_token[mask == 2]

        batch["input_ids"] = input_ids

        # define mask (which tokens should be predicted)
        # batch["attention_mask"] = mask != 0

        return batch
