from nltk import tokenize
from transformers import PreTrainedTokenizerBase
from datasets import Dataset


def split(text: str):
    return tokenize.sent_tokenize(text)


class ContextLengthSplitter:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 4):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(self, dataset: Dataset, context_length: int = 512, num_samples: int = 10_000):
        return dataset.map(
                self.split_sample_by_context_length,
                batched=True,
                batch_size=self.batch_size,
                remove_columns=["id", "url", "title"],
                fn_kwargs={"context_length": context_length},
                desc=f"Splitting by context length",
                num_proc=8
            )

    def split_sample_by_context_length(self, batch, context_length: int):
        """
        splits samples into blocks with a length of up to context_length
        each sample must contain more than 1 sentence because it must be split for further processing
        """

        result = {"text": []} #"length": []

        # count length of used text and length of text thrown away

        for sample in batch["text"]:
            # splitted_text = sample["text"][0].split(".")
            splitted_text = split(sample)
            tokenized = self.tokenizer(splitted_text, truncation=True)
            tokenized["text"] = splitted_text

            current_sample_text = []
            running_length = 0
            current_num_sentences = 0

            for index, (text, input_ids) in enumerate(zip(tokenized["text"], tokenized["input_ids"])):
                input_num_tokens = len(input_ids)
                # -3 because 1x [CLS], 2x[SEP]
                if running_length + input_num_tokens > context_length - 3:
                    new_sample = " ".join(current_sample_text)
                    if current_num_sentences > 1:
                        result["text"].append(new_sample)
                        # result["length"].append(running_length)
                    current_sample_text = []
                    running_length = 0
                    current_num_sentences = 0

                current_sample_text.append(text)
                running_length += input_num_tokens
                current_num_sentences += 1

            # add the last sentence
            context_length_upper_fifth = round(context_length * (4 / 5))
            if running_length > context_length_upper_fifth and current_num_sentences > 0:
                new_sample = " ".join(current_sample_text)
                result["text"].append(new_sample)
                # result["length"].append(running_length)

        return result