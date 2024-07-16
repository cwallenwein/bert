from nltk import tokenize
from datasets import enable_progress_bars, disable_progress_bars
from contextlib import contextmanager


def split_into_sentences(text: str):
    return tokenize.sent_tokenize(text)


def append_to_batch(batch: dict, sample: dict):
    assert batch.keys() == sample.keys()

    for key in batch.keys():
        batch[key].append(sample[key])
    return batch


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
