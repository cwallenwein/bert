import math
import time
import torch
import random
from nltk import tokenize
from itertools import product
from transformers import PreTrainedTokenizerBase
from datasets import interleave_datasets, Dataset, load_dataset, load_from_disk
from data.load import get_dataset_path
from data.context_length_splitter import ContextLengthSplitter


def is_power_of_2(number: int):
    exponent = math.log2(number)
    return int(exponent) == exponent


def split(text: str):
    return tokenize.sent_tokenize(text)


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.total_time_in_sec = 0
        self.mask_token_id = self.tokenizer.mask_token_id

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        tokenized = self.tokenizer(*args, **kwargs)
        end_time = time.time()
        self.total_time_in_sec += end_time - start_time
        return tokenized

    def __len__(self):
        return len(self.tokenizer)


class MaskedLanguageModelingPreprocessor:

    def __init__(self,
                 context_length: int = 512,
                 batch_size: int = 8,
                 p_mask: float = 0.15,
                 p_replacement_mask: float = 0.8,
                 p_replacement_random: float = 0.1,
                 ):

        assert 0 <= p_mask <= 1, "p_mask must be between 0 and 1"
        assert 0 <= p_replacement_mask + p_replacement_random <= 1, "p_replacement_mask + p_replacement_random must be between 0 and 1"
        self.p_mask = p_mask
        self.p_replacement_mask = p_replacement_mask
        self.p_replacement_random = p_replacement_random
        self.p_replacement_unchanged = 1 - p_replacement_mask - p_replacement_random

        self.batch_size = batch_size

        assert is_power_of_2(context_length), "context_length must be a power of 2"
        self.context_length = context_length

    def __call__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        num_samples: int = 10_000
    ):

        tokenizer = TokenizerWrapper(tokenizer=tokenizer)
        tokenizer.tokenizer.model_max_length = 10 ** 10

        self.is_next_splitter = IsNextSplitter()
        self.not_next_splitter = NotNextSplitter(tokenizer=tokenizer)

        # dataset.cleanup_cache_files()

        dataset = self.split_by_context_length(dataset, tokenizer=tokenizer)
        # apply nsp
        # apply mlm

        # dataset = self.prepare_masked_language_modeling(dataset, tokenizer=tokenizer, num_samples=num_samples)

        return dataset

    def split_by_context_length(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase | TokenizerWrapper):
        dataset_dir = get_dataset_path(split_by_context_length=True, context_length=self.context_length)

        if dataset_dir.exists():
            dataset = load_from_disk(str(dataset_dir))
            print(f"Loaded dataset with max {self.context_length} tokens from file.")
        else:
            context_length_splitter = ContextLengthSplitter(tokenizer=tokenizer)
            dataset = context_length_splitter(dataset, self.context_length)
            dataset.save_to_disk(str(dataset_dir))

        return dataset

    def prepare_masked_language_modeling(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase | TokenizerWrapper,
        num_samples: int = 10_000,
        batch_size: int = 8,
        save: bool = True,
        out_dir: str = "mlm_dataset",
        overwrite=False
    ):
        dataset_dir = get_dataset_path(mlm_and_nsp=True, context_length=self.context_length)

        if dataset_dir.exists():
            dataset = load_from_disk(str(dataset_dir))
            print("Loaded MLM dataset from file.")
            return dataset

        assert batch_size % 2 == 0, "batch_size must be even"

        dataset_length = len(dataset)
        dataset = dataset.shuffle(seed=42)

        # get indices for all random samples
        # random_indices = []
        # for i in range(num_samples // len(dataset)):
        #     random_indices += random.sample(range(dataset_length), dataset_length)
        # if num_samples % len(dataset) > 0:
        #     random_indices += random.sample(range(dataset_length), num_samples % len(dataset))
        # dataset = dataset.select(random_indices)

        # for batch in dataset:
        #     if random.random() < 0.5:
        #         is_next = self.is_next_splitter(batch)
                # results in 1 sample each
            # else:
            #     not_next = self.not_next_splitter(batch)
                # results in 2 samples each

            # sample from dataset until num_samples is reached
            #   pass each batch either to is_next or not_next

        assert dataset_length > num_samples

        # split data into is_next and not_next
        is_next = dataset.select(range(0, num_samples, 2))
        is_not_next = dataset.select(range(1, num_samples, 2))

        # process is_next and not_next
        is_next = self.is_next_splitter(is_next)
        is_not_next = self.not_next_splitter(is_not_next, context_length=self.context_length)

        # concatenate is_next and not_next
        tokenized_bert_dataset = interleave_datasets([is_next, is_not_next], probabilities=[0.5, 0.5], seed=42)

        # tokenize and mask
        masked_lm_data_collator = MaskedLMDataCollator(
            context_length=self.context_length,
            batch_size=self.batch_size,
            tokenizer=tokenizer,
            p_mask=self.p_mask,
            p_replacement_mask=self.p_replacement_mask,
            p_replacement_random=self.p_replacement_random,
            p_replacement_unchanged=self.p_replacement_unchanged
        )

        masked_bert_dataset = masked_lm_data_collator(tokenized_bert_dataset)
        shuffled_and_masked_bert_dataset = masked_bert_dataset.shuffle(seed=42)

        shuffled_and_masked_bert_dataset.save_to_disk(str(dataset_dir))

        return shuffled_and_masked_bert_dataset


class IsNextSplitter:

    def __call__(self, dataset):
        return dataset.map(
            self.split_is_next,
            batched=True,
            batch_size=1,
            desc=f"Splitting is next",
            remove_columns=dataset.column_names
        )

    @staticmethod
    def split_is_next(batch):

        splitted_texts = []

        for text in batch["text"]:

            all_splits = split(text)

            if len(all_splits) < 2:
                return {"text": []}

            split_index = random.randint(1, len(all_splits) - 1)
            left_side = " ".join(all_splits[:split_index])
            right_side = " ".join(all_splits[split_index:])
            # assert len(left_side) > 0 and len(right_side) > 0
            splitted_texts.append((left_side, right_side))

        return {"text": splitted_texts}


class NotNextSplitter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, dataset: Dataset, context_length: int):
        not_next = dataset.map(
            self.split_not_next,
            batched=True,
            batch_size=2,
            drop_last_batch=True,
            fn_kwargs={"context_length": context_length},
            desc=f"Processing is not next",
            remove_columns=dataset.column_names
        )

        return not_next

    def split_not_next(self, batch, context_length: int):
        batch = batch["text"]
        assert len(batch) % 2 == 0

        result = []

        for i in range(0, len(batch), 2):
            text1, text2 = batch[i:i + 2]

            # assert len(text1) != 0
            # assert len(text2) != 0

            sentences1 = split(text1)
            sentences2 = split(text2)

            split_options_text1 = range(1, len(sentences1))
            split_options_text2 = range(1, len(sentences2))

            sentence_lengths_text1 = [len(tokenized_sentence) for tokenized_sentence in
                                      self.tokenizer(sentences1)["input_ids"]]
            sentence_lengths_text2 = [len(tokenized_sentence) for tokenized_sentence in
                                      self.tokenizer(sentences2)["input_ids"]]

            all_split_options = product(split_options_text1, split_options_text2)

            possible_combinations = []
            for split_idx1, split_idx2 in all_split_options:
                text1_1st_half = Half(sentences1[:split_idx1], sum(sentence_lengths_text1[:split_idx1]))
                text1_2nd_half = Half(sentences1[split_idx1:], sum(sentence_lengths_text1[split_idx1:]))
                text2_1st_half = Half(sentences2[:split_idx2], sum(sentence_lengths_text2[:split_idx2]))
                text2_2nd_half = Half(sentences2[split_idx2:], sum(sentence_lengths_text2[split_idx2:]))

                combinations = (
                    (Sample(text1_1st_half, text2_1st_half), Sample(text1_2nd_half, text2_2nd_half)),
                    (Sample(text1_1st_half, text2_2nd_half), Sample(text1_2nd_half, text2_1st_half))
                )

                for combination in combinations:
                    if combination[0].total_length <= context_length - 3 and \
                        combination[1].total_length <= context_length - 3:
                        possible_combinations.append(combination)

            if len(possible_combinations) > 1:
                chosen_combination = random.choice(possible_combinations)

                result.append((chosen_combination[0].text1, chosen_combination[0].text2))
                result.append((chosen_combination[1].text1, chosen_combination[1].text2))

        return {"text": result}


class MaskedLMDataCollator:

    def __init__(self,
                 context_length: int,
                 batch_size: int,
                 tokenizer: PreTrainedTokenizerBase,
                 p_mask: float = 0.15,
                 p_replacement_mask: float = 0.8,
                 p_replacement_random: float = 0.1,
                 p_replacement_unchanged=0.1
                 ):
        self.context_length = context_length
        self.batch_size = batch_size
        self.p_mask = p_mask
        self.p_replacement_mask = p_replacement_mask
        self.p_replacement_random = p_replacement_random
        self.p_replacement_unchanged = p_replacement_unchanged
        self.tokenizer = tokenizer

        assert self.p_replacement_mask + self.p_replacement_random + self.p_replacement_unchanged == 1
        self.num_mask_tokens = round(context_length * p_mask)
        self.vocab_size = len(tokenizer)

    def __call__(self, dataset) -> Dataset:
        return dataset \
            .with_format("torch") \
            .map(
            self.mask_and_tokenize,
            batched=True,
            batch_size=self.batch_size,
            num_proc=8,
            desc=f"Masking dataset",
        )

    def mask_and_tokenize(self, batch):
        tokenized_batch = self.tokenize_batch(batch=batch)
        masked_batch = self.mask_batch(tokenized_batch)
        return masked_batch

    def tokenize_batch(self, batch: dict):
        # Tokenize batch
        text = batch["text"]
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.context_length,
            return_special_tokens_mask=True
        )
        batch["text"] = text
        return batch

    def mask_batch(self, batch: dict):
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

        batch["masked_token_mask"] = ((mask == 1) | (mask == 2) | (mask == 3)).int()

        # replace values according to mask
        input_ids = batch["input_ids"]

        input_ids[mask == 1] = self.tokenizer.mask_token_id
        random_token = torch.randint_like(input_ids, self.vocab_size)
        input_ids[mask == 2] = random_token[mask == 2]

        batch["input_ids"] = input_ids

        # define labels
        batch["labels"] = input_ids.detach()

        # define mask (which tokens should be predicted)
        # batch["attention_mask"] = mask != 0

        return batch


class Half:
    def __init__(self, text: str, length: int):
        self.text = text
        self.length = length


class Sample:
    def __init__(self, half1: Half, half2: Half):
        self.text1 = " ".join(half1.text)
        self.text2 = " ".join(half2.text)
        self.total_length = half1.length + half2.length
