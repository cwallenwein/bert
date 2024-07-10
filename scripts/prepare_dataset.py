# Usage: python scripts/prepare_dataset.py --context_length=128

import argparse
from data.bert_dataset import BertDataset


def prepare_dataset(context_length: int = 128):
    return BertDataset.prepare(context_length)


parser = argparse.ArgumentParser(description="Prepare dataset")
parser.add_argument("--context_length", type=int, help="Context length to prepare")
args = parser.parse_args()
prepare_dataset(**args.__dict__)
