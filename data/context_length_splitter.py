from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from data.util import split_into_sentences


class ContextLengthSplitter:
    """

    lowest possible block id: 0
    invalid block: -1
    """

    # TODO: replace blocks by (start,finish)-tuples

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(
        self,
        dataset: Dataset,
        context_length: int = 128,
        batch_size: int = 4
    ):
        effective_context_length = context_length - 3
        context_length_80_pct = round(context_length * 0.8)
        return dataset.map(
            self.split_sample_by_context_length,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "min_context_length": context_length_80_pct,
                "max_context_length": effective_context_length,
            },
            input_columns=["text"],
            desc=f"Splitting samples into blocks of {context_length} tokens",
            num_proc=6
        )

    def split_sample_by_context_length(self, text_batch, min_context_length: int, max_context_length: int):
        new_input_ids = []
        for sample in text_batch:
            # add space because it is removed during split
            sentences = [" " + sentence for sentence in split_into_sentences(sample)]
            sample_input_ids = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
            sentence_lengths = [len(sentence) for sentence in sample_input_ids]
            block_ids: list[int] = self._get_blocks(
                sentence_lengths, min_context_length=min_context_length, max_context_length=max_context_length
            )
            new_input_ids.extend(
                self._separate_input_ids_into_blocks(sample_input_ids, block_ids)
            )
        assert all(
            min_context_length <= sum(len(ids) for ids in input_ids) <= max_context_length
            for input_ids in new_input_ids
        )
        return {"input_ids": new_input_ids}

    @staticmethod
    def _separate_input_ids_into_blocks(
        sample_input_ids,
        block_ids,
    ):
        new_input_ids = []
        # combine results inplace
        current_block_id = -1
        for sentence_block_id, sentence_input_ids in zip(block_ids, sample_input_ids):
            # each value is a sentence
            if sentence_block_id == -1:
                continue
            elif sentence_block_id == current_block_id:
                # print("add to existing sample")
                new_input_ids[sentence_block_id].append(sentence_input_ids)
            else:
                # print("create new sample")
                current_block_id = sentence_block_id
                new_input_ids.append([sentence_input_ids])
        return new_input_ids

    @staticmethod
    def _get_blocks(
        sentence_lengths: list[int],
        min_context_length: int = 102,
        max_context_length: int = 125,
    ) -> list[int]:
        """
        Split a list of sentence lengths into blocks of sentences that fit into a context length.
        """

        blocks = []
        current_block = []
        current_block_length = 0
        current_block_id = 0
        for sentence_length in sentence_lengths:
            if sentence_length > max_context_length:
                # handle previous sentences in block
                if len(current_block) >= 2 and current_block_length >= min_context_length:
                    blocks.extend(current_block)
                    current_block_id += 1
                else:
                    blocks.extend([-1] * len(current_block))

                # handle this sentence
                blocks.append(-1)
                current_block = []
                current_block_length = 0

            elif current_block_length + sentence_length > max_context_length:
                # handle previous sentences in block

                # there must be at least two blocks / sentences in a sample
                if len(current_block) >= 2 and current_block_length >= min_context_length:
                    blocks.extend(current_block)
                    current_block_id += 1
                else:
                    blocks.extend([-1] * len(current_block))

                # handle this sentence
                current_block = [current_block_id]
                current_block_length = sentence_length
            else:
                # increase
                current_block.append(current_block_id)
                current_block_length += sentence_length

        if len(current_block) >= 2 and current_block_length >= min_context_length:
            blocks.extend(current_block)
        else:
            blocks.extend([-1] * len(current_block))

        return blocks


def main():
    pass


if __name__ == "__main__":
    main()
