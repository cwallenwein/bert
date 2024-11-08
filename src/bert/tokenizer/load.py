from pathlib import Path

from transformers import AutoTokenizer

from bert.utils import get_tokenizers_base_dir

"""
loads local tokenizers and HF tokenizers

tokenizer_id: raw_dataset_name/tokenizer_name
"""


def load_tokenizer(tokenizer_id_or_name: str):
    try:
        tokenizer = load_local_tokenizer(tokenizer_id_or_name)
    except FileNotFoundError:
        tokenizer = load_hf_tokenizer(tokenizer_id_or_name)
    finally:
        return tokenizer


def load_hf_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_local_tokenizer(tokenizer_id: str):
    tokenizers_base_dir = get_tokenizers_base_dir()
    tokenizer_path = Path(tokenizers_base_dir) / tokenizer_id
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Can't find tokenizer with tokenizer_id {tokenizer_id} locally"
        )

    return AutoTokenizer.from_pretrained(tokenizer_path)
