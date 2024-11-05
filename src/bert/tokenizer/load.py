from pathlib import Path

from transformers import AutoTokenizer

from bert.utils import get_tokenizers_base_dir


def load(raw_dataset_name: str, tokenizer_name: str):
    tokenizers_base_dir = get_tokenizers_base_dir()
    tokenizer_path = Path(tokenizers_base_dir) / raw_dataset_name / tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer
