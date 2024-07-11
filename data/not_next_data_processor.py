from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import random
from itertools import product


@dataclass
class NewSampleInfo:
    input_ids_first_half: list
    input_ids_second_half: list
    indices_first_half: slice
    indices_second_half: slice
    sentence_lengths_first_half: list
    sentence_lengths_second_half: list

    def __len__(self):
        len_first_half = sum(self.sentence_lengths_first_half[self.indices_first_half])
        len_second_half = sum(self.sentence_lengths_second_half[self.indices_second_half])
        return len_first_half + len_second_half


class NotNextDataProcessor:

    def __init__(self, tokenizer: PreTrainedTokenizerFast, dataset: Dataset):
        assert dataset.column_names == ["input_ids"]
        self.dataset = dataset
        self.column_names = dataset.column_names
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.num_not_possible = 0
        self.num_batches = 0

    def __call__(self, context_length: int = 128, verbose: bool = True):
        if verbose:
            print("not next splitting fails <=40% of the time -> make sure to generate enough samples!")
        batch_size = 2
        assert batch_size == 2
        dataset = self.dataset.map(
            self.split_not_next,
            batched=True,
            batch_size=batch_size,
            desc=f"Apply not next split",
            num_proc=6,
            input_columns=["input_ids"],
            fn_kwargs={"context_length": context_length},
            drop_last_batch=True
        ).with_format(
            "torch"
        )
        return dataset

    def split_not_next(self, input_ids, context_length: int = 128):
        self.num_batches += 1

        possible_cross_combinations = self._get_possible_cross_combinations(input_ids, context_length=context_length)

        if len(possible_cross_combinations) > 0:
            new_sample1_info, new_sample2_info = random.choice(possible_cross_combinations)
            return self._create_new_samples_from_new_sample_info(
                [new_sample1_info, new_sample2_info], context_length=context_length
            )
        else:
            self.num_not_possible += 1
            return {
                "input_ids": [],
                "attention_mask": [],
                "token_type_ids": []
            }

    def _create_new_samples_from_new_sample_info(
        self, sample_infos: list[NewSampleInfo], context_length: int = 128
    ):
        new_batch = dict()
        input_ids = self._create_input_ids_from_new_sample_infos(sample_infos, context_length)
        new_batch["input_ids"] = input_ids
        new_batch["token_type_ids"] = self._create_token_type_ids(input_ids)
        new_batch["attention_mask"] = self._create_attention_mask(input_ids)
        return new_batch

    def _create_input_ids_from_new_sample_infos(self, sample_infos: list[NewSampleInfo], context_length: int = 128):
        return [
            self.create_input_ids_from_new_sample_info(sample_info, context_length)
            for sample_info in sample_infos
        ]

    def create_input_ids_from_new_sample_info(self, sample_info: NewSampleInfo, context_length: int = 128):
        new_sample = [self.cls_token_id]
        for sentence in sample_info.input_ids_first_half[sample_info.indices_first_half]:
            new_sample.extend(sentence)
        new_sample.append(self.sep_token_id)
        for sentence in sample_info.input_ids_second_half[sample_info.indices_second_half]:
            new_sample.extend(sentence)
        new_sample.append(self.sep_token_id)
        assert len(new_sample) <= context_length, f"new_sample has length: {len(new_sample)}"
        new_sample.extend([self.pad_token_id] * (context_length - len(new_sample)))
        return new_sample

    def _create_token_type_ids(self, input_ids_batch: list):
        token_type_ids_batch = []
        for input_ids in input_ids_batch:
            sep_index = input_ids.index(self.sep_token_id)
            token_type_ids = [0] * (sep_index + 1) + [1] * (len(input_ids) - sep_index - 1)
            token_type_ids_batch.append(token_type_ids)
        return token_type_ids_batch

    def _create_attention_mask(self, input_ids_batch):
        attention_mask_batch = []
        for input_ids in input_ids_batch:
            attention_mask = [0 if input_id == self.pad_token_id else 1 for input_id in input_ids]
            attention_mask_batch.append(attention_mask)
        return attention_mask_batch

    def _get_possible_cross_combinations(self, input_ids: list, context_length: int = 128):
        sample_a, sample_b = input_ids
        sample_a_num_sentences = len(sample_a)
        sample_b_num_sentences = len(sample_b)
        sample_a_len_sentences = [len(sentence) for sentence in sample_a]
        sample_b_len_sentences = [len(sentence) for sentence in sample_b]

        possible_combinations = []
        while True:
            for split_idx_a, split_idx_b in product(
                range(1, sample_a_num_sentences - 1),
                range(1, sample_b_num_sentences - 1)
            ):
                # combine first half of sample_a and first half of sample_b
                new_sample1 = NewSampleInfo(
                    input_ids_first_half=sample_a,
                    input_ids_second_half=sample_b,
                    indices_first_half=slice(0, split_idx_a),
                    indices_second_half=slice(0, split_idx_b),
                    sentence_lengths_first_half=sample_a_len_sentences,
                    sentence_lengths_second_half=sample_b_len_sentences
                )
                new_sample2 = NewSampleInfo(
                    input_ids_first_half=sample_a,
                    input_ids_second_half=sample_b,
                    indices_first_half=slice(split_idx_a, sample_a_num_sentences),
                    indices_second_half=slice(split_idx_b, sample_b_num_sentences),
                    sentence_lengths_first_half=sample_a_len_sentences,
                    sentence_lengths_second_half=sample_b_len_sentences
                )

                if len(new_sample1) <= context_length - 3 and len(new_sample2) <= context_length - 3:
                    possible_combinations.append([new_sample1, new_sample2])

                # combine first half of sample1 and second half of sample2
                new_sample1 = NewSampleInfo(
                    input_ids_first_half=sample_a,
                    input_ids_second_half=sample_b,
                    indices_first_half=slice(0, split_idx_a),
                    indices_second_half=slice(split_idx_b, sample_b_num_sentences),
                    sentence_lengths_first_half=sample_a_len_sentences,
                    sentence_lengths_second_half=sample_b_len_sentences
                )

                new_sample2 = NewSampleInfo(
                    input_ids_first_half=sample_a,
                    input_ids_second_half=sample_b,
                    indices_first_half=slice(split_idx_a, sample_a_num_sentences),
                    indices_second_half=slice(0, split_idx_b),
                    sentence_lengths_first_half=sample_a_len_sentences,
                    sentence_lengths_second_half=sample_b_len_sentences
                )

                if len(new_sample1) <= context_length - 3 and len(new_sample2) <= context_length - 3:
                    possible_combinations.append([new_sample1, new_sample2])

            for possible_combination in possible_combinations:
                sample_info1, sample_info2 = possible_combination
                sample1 = self.create_input_ids_from_new_sample_info(sample_info1, context_length)
                sample2 = self.create_input_ids_from_new_sample_info(sample_info2, context_length)
                if len(sample1) > context_length or len(sample2) > context_length:
                    raise ValueError(f"sample1: {len(sample1)}, sample2: {len(sample2)}")

            if len(possible_combinations) > 0:
                return possible_combinations
            if sample_a_num_sentences <= 2 and sample_b_num_sentences <= 2:
                return possible_combinations
            if sample_a_num_sentences > sample_b_num_sentences:
                sample_a_num_sentences -= 1
            else:
                sample_b_num_sentences -= 1
