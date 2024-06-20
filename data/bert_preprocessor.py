from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase
from data.mlm_preprocessor import MaskedLanguageModelingPreprocessor


class BertPreprocessor:

    def __init__(self,
                 small_context_pct: float = 0.9,
                 small_context_length: int = 128,
                 big_context_length: int = 512,
                 batch_size: int = 8,
                 p_mask: float = 0.15,
                 p_replacement_mask: float = 0.8,
                 p_replacement_random: float = 0.1
                 ):

        assert 0 <= small_context_pct <= 1, "small_context_pct must be between 0 and 1"
        self.small_context_pct: float = small_context_pct
        self.big_context_pct: float = 1 - small_context_pct

        assert small_context_length < big_context_length, "small_context_length must be smaller than big_context_length"

        self.small_context_preprocessor = MaskedLanguageModelingPreprocessor(
            context_length=small_context_length,
            batch_size=batch_size,
            p_mask=p_mask,
            p_replacement_mask=p_replacement_mask,
            p_replacement_random=p_replacement_random,
        )

        self.big_context_preprocessor = MaskedLanguageModelingPreprocessor(
            context_length=big_context_length,
            batch_size=batch_size,
            p_mask=p_mask,
            p_replacement_mask=p_replacement_mask,
            p_replacement_random=p_replacement_random,
        )

    def __call__(self,
                 dataset: Dataset,
                 tokenizer: PreTrainedTokenizerBase,
                 num_samples: int = 10_000,
                 # save: bool = True,
                 # out_dir: str = "mlm_dataset",
                 # overwrite=False
                 ):

        # total_num_samples = len(dataset)
        # cutoff = int(total_num_samples * self.small_context_pct)
        # dataset = dataset.shuffle(seed=42)
        # small_context_dataset = dataset.select(range(0, cutoff))
        # big_context_dataset = dataset.select(range(cutoff, total_num_samples))

        num_samples_small_context = int(num_samples * self.small_context_pct)
        num_samples_big_context = num_samples - num_samples_small_context

        small_context_dataset = self.small_context_preprocessor(
            dataset,
            tokenizer=tokenizer,
            num_samples=num_samples_small_context,
            # save=save,
            # out_dir=out_dir,
            # overwrite=overwrite
        )
        big_context_dataset = self.big_context_preprocessor(
            dataset,
            tokenizer=tokenizer,
            num_samples=num_samples_big_context,
            # save=save,
            # out_dir=out_dir,
            # overwrite=overwrite
        )

        dataset = concatenate_datasets([small_context_dataset, big_context_dataset])

        return dataset
