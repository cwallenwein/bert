from pathlib import Path

from bert.utils import get_datasets_raw_dir
from datasets import load_dataset, load_from_disk


def download_hf_dataset(
    dataset_name: str,
    *hf_load_dataset_args,
    **hf_load_dataset_kwargs,
):
    dataset = load_dataset(*hf_load_dataset_args, **hf_load_dataset_kwargs)

    raw_datasets_dir = get_datasets_raw_dir()
    dataset_path = Path(raw_datasets_dir) / dataset_name
    dataset.save_to_disk(str(dataset_path))


def load_local_hf_dataset(dataset_name: str):
    raw_datasets_dir = get_datasets_raw_dir()
    dataset_path = Path(raw_datasets_dir) / dataset_name
    dataset = load_from_disk(str(dataset_path))
    return dataset
