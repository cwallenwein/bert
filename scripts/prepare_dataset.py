# Usage: python scripts/prepare_dataset.py --num_samples=10

import argparse
from data import BertPreprocessor
from data.load import get_dataset
from transformers import BertTokenizer


def prepare_dataset(num_samples: int = 100_000):
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    dataset = get_dataset(raw=True)

    bert_preprocessor = BertPreprocessor()
    preprocessed_dataset = bert_preprocessor(dataset, tokenizer, num_samples=num_samples)

    return preprocessed_dataset


parser = argparse.ArgumentParser(description="Prepare dataset")
parser.add_argument("--num_samples", type=int, help="Number of samples to prepare")
args = parser.parse_args()
prepare_dataset(**args.__dict__)