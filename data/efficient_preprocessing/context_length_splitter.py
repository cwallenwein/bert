from nltk import tokenize
from transformers import PreTrainedTokenizerBase
from datasets import Dataset


def split(text: str):
    return tokenize.sent_tokenize(text)


# TODO: replace blocks by (start,finish)-tuples
class ContextLengthSplitter:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 4):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(self, dataset: Dataset, context_length: int = 512, lists_of_sentences: bool = False):
        self.LOWEST_BLOCK_ID = 0
        self.INVALID_BLOCK_ID = -1

        dataset.cleanup_cache_files()
        dataset = dataset.map(
            self.split_sample_by_context_length,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["id", "url", "title"],
            fn_kwargs={"context_length": context_length, "lists_of_sentences": lists_of_sentences},
            desc=f"Splitting by context length (2)",
            # num_proc=8
        )
        return dataset

    """
    [["this is a sentences", "this is another sentence"], ["this is a third sentence", "this is a fourth sentence"]]
    [["this is a sentences. this is another sentence"], ["this is a third sentence. this is a fourth sentence"]]

    [[S,1], [S,2]], [[S,1], [S,2]]: sentences are in separate list elements; try first; this would make shuffling easier as well
    [[S,1,S,2], [S,1,S,2]]: all sentences are concatenated into a single sample
    """

    def split_sample_by_context_length(self, batch, context_length: int, lists_of_sentences: bool):
        start_block_id = 0
        result = {"text": [], "input_ids": [], "token_type_ids": [], "attention_mask": []}
        for sample in batch["text"]:
            text = split(sample)
            # add space back because it was removed during split
            text = [" " + sentence for sentence in text]
            tokenized = self.tokenizer(text, add_special_tokens=False)
            tokenized["text"] = text
            sentence_lengths = [len(sentence) for sentence in tokenized["input_ids"]]
            blocks: list[int] = self.get_blocks(sentence_lengths, context_length, start_id=start_block_id)
            start_block_id = max(blocks) + 1
            self.update_result(result, tokenized, blocks, lists_of_sentences=lists_of_sentences)
        return result

    def update_result(self, result, tokenized, blocks, lists_of_sentences=False):
        # if max(blocks) == 0:
        #     return

        for key, values in tokenized.items():
            # combine results inplace
            current_block_id = -10
            for block_id, value in zip(blocks, values):
                # each value is a sentence
                if block_id == current_block_id:
                    # print("add to existing sample")
                    if lists_of_sentences:
                        result[key][block_id].append(value)
                    else:
                        result[key][block_id] += value
                elif block_id != self.INVALID_BLOCK_ID:
                    # print("create new sample")
                    current_block_id = block_id
                    if lists_of_sentences:
                        result[key].append([value])
                    else:
                        result[key].append(value)

    def get_blocks(self, sentence_lengths: list[int], context_length: int = 512, start_id: int = 0) -> list[int]:
        assert start_id >= 0
        effective_context_length = context_length - 3
        context_length_80_pct = round(context_length * 0.8)

        # get_blocks([1,2,3]) = [1,1,2]
        # get_blocks([1,4,2,3]) = [-1,-1,1,2]
        # get_blocks([2,4,2,3]) = [1,-1,2,3]
        # get_blocks([1,2,3,4]) = [1,1,2,-1]

        blocks = []
        current_block = []
        current_block_length = 0
        current_block_id = start_id
        for sentence_length in sentence_lengths:
            if sentence_length > effective_context_length:
                # handle previous sentences in block
                if current_block_length >= context_length_80_pct:
                    blocks.extend(current_block)
                    current_block_id += 1
                else:
                    blocks.extend([self.INVALID_BLOCK_ID] * len(current_block))

                # handle this sentence
                blocks.append(self.INVALID_BLOCK_ID)
                current_block = []
                current_block_length = 0

            elif current_block_length + sentence_length > effective_context_length:
                # handle previous sentences in block
                if len(current_block) > 1:
                    blocks.extend(current_block)
                    current_block_id += 1
                else:
                    blocks.extend([self.INVALID_BLOCK_ID] * len(current_block))

                # handle this sentence
                current_block = [current_block_id]
                current_block_length = sentence_length
            else:
                # increase
                current_block.append(current_block_id)
                current_block_length += sentence_length

        if current_block_length >= context_length_80_pct:
            blocks.extend(current_block)
        else:
            blocks.extend([self.INVALID_BLOCK_ID] * len(current_block))

        assert len(sentence_lengths) == len(blocks)
        return blocks

# from transformers import BertTokenizer
# from data.load import get_dataset
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# dataset = get_dataset(raw=True)
# dataset.cleanup_cache_files()
#
# c = ContextLengthSplitter(tokenizer=tokenizer, batch_size=2)
# d = c(dataset=dataset.select(range(100)), lists_of_sentences=True, context_length=128)
