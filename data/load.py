from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset


def get_dataset_path(raw: bool = False, split_by_context_length: bool = False, mlm_and_nsp: bool = False, context_length: int = None) -> Path:
    # make sure only one argument is True
    assert sum([raw, split_by_context_length, mlm_and_nsp]) == 1
    # raw data doesn't have context length
    assert not (raw and context_length)

    dataset_name = "wikipedia"

    base_path = Path(__file__).parent / "datasets"
    if raw:
        dataset_path = base_path / "raw" / dataset_name
    elif split_by_context_length:
        dataset_path = base_path / "split_by_context_length" / dataset_name
    elif mlm_and_nsp:
        dataset_path = base_path / "mlm_and_nsp" / dataset_name
    else:
        raise ValueError("Invalid arguments")

    if context_length:
        dataset_path = dataset_path / str(context_length)

    return dataset_path


def get_dataset(raw: bool = False, split_by_context_length: bool = False, mlm_and_nsp: bool = False, context_length: int = None) -> Dataset | DatasetDict:
    # make sure only one argument is True
    assert sum([raw, split_by_context_length, mlm_and_nsp]) == 1
    # raw data doesn't have context length
    assert not (raw and context_length)

    path = get_dataset_path(raw, split_by_context_length, mlm_and_nsp, context_length)

    if raw:
        if path.exists():
            return Dataset.load_from_disk(str(path))
        else:
            path.mkdir(parents=True, exist_ok=True)
            wikipedia = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]
            wikipedia.save_to_disk(str(path))
            return wikipedia

    if context_length:
        return Dataset.load_from_disk(str(path))
    else:
        datasets = {}
        for context_length_dir in path.iterdir():
            context_length = int(context_length_dir.stem)
            datasets[context_length] = Dataset.load_from_disk(str(context_length_dir))
        datasets = DatasetDict(datasets)
        return datasets

