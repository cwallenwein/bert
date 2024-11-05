from pathlib import Path

from transformers import PreTrainedTokenizer

from bert.utils import get_tokenizers_base_dir


def save_tokenizer(
    tokenizer: PreTrainedTokenizer,
    raw_dataset_name: str,
    tokenizer_name: str,
):
    """
    Save the transformers.PreTrainedTokenizer
    """
    tokenizers_base_dir = get_tokenizers_base_dir()
    tokenizer_path = Path(tokenizers_base_dir) / raw_dataset_name / tokenizer_name
    tokenizer.save_pretrained(tokenizer_path)
