from nltk import tokenize
from datasets import enable_progress_bars, disable_progress_bars
from contextlib import contextmanager
import math
from typing import Dict, List


def is_power_of_2(number: int):
    return math.log2(number).is_integer()


def split_into_sentences(text: str):
    return tokenize.sent_tokenize(text)


def append_to_batch(batch: dict, sample: dict):
    assert batch.keys() == sample.keys()

    for key in batch.keys():
        batch[key].append(sample[key])
    return batch


def add_early_stopping(examples: Dict[str, List], rank: int, map_function, max_samples: int, sample_count: List[int], empty_sample):
    """
    Use like this:

        from functools import partial

        tokenize_with_early_stopping = partial(
            add_early_stopping,
            map_function=tokenize,
            max_samples=150,
            current_samples={"count": 0},
        )
    """
    rank = 0 if rank is None else rank
    mapped_examples = map_function(examples)

    batch_size = len(next(iter(mapped_examples.values())))

    # test if sample should be truncated
    remaining_samples = max_samples - sample_count[rank]
    if remaining_samples <= 0:
        # TODO: empty sample
        # return {key: [] for key in mapped_examples}
        return empty_sample

    if batch_size > remaining_samples:
        # truncate
        for key in mapped_examples:
            mapped_examples[key] = mapped_examples[key][:remaining_samples]
        sample_count[rank] = max_samples
    else:
        sample_count[rank] += batch_size

    print(f"sample_count[{rank}]", sample_count[rank])

    return mapped_examples


@contextmanager
def set_progress_bar(enable_progress_bar: bool):
    try:
        if enable_progress_bar:
            enable_progress_bars()
        else:
            disable_progress_bars()
        yield
    finally:
        enable_progress_bars()
