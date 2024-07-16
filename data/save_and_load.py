from pathlib import Path
from transformers import BertTokenizerFast
from datasets import Dataset, DatasetDict, load_dataset
import shutil


def get_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = 2 ** 18
    return tokenizer


def get_dataset_path(
    raw: bool = False,
    split_by_context_length: bool = False,
    preprocessed: bool = False,
    preprocessed_name: str = None,
    context_length: int = None,
) -> Path:
    if raw:
        assert not context_length and not split_by_context_length and not preprocessed
    if split_by_context_length:
        assert not raw and not preprocessed
    if preprocessed:
        assert preprocessed_name and not raw and not split_by_context_length

    tokenizer = get_tokenizer()
    tokenizer_name = tokenizer.name_or_path.replace("/", "-")
    dataset_name = "wikipedia"

    base_path = Path(__file__).parent / "datasets"
    if raw:
        dataset_path = base_path / "raw" / dataset_name
    elif split_by_context_length:
        dataset_path = base_path / "split_by_context_length" / dataset_name / tokenizer_name
        if context_length:
            dataset_path = dataset_path / str(context_length)
    elif preprocessed:
        dataset_path = base_path / "preprocessed" / preprocessed_name
        if context_length:
            dataset_path = dataset_path / str(context_length)
    else:
        raise ValueError("Invalid arguments")

    return dataset_path


def get_dataset(
    raw: bool = False,
    split_by_context_length: bool = False,
    preprocessed: bool = False,
    preprocessed_name: str = None,
    context_length: int = None
) -> Dataset | DatasetDict:
    # make sure only one argument is True
    assert sum([raw, split_by_context_length, preprocessed]) == 1
    # wikipedia data doesn't have context length
    assert not (raw and context_length)

    path = get_dataset_path(
        raw=raw,
        split_by_context_length=split_by_context_length,
        preprocessed=preprocessed,
        preprocessed_name=preprocessed_name,
        context_length=context_length
    )

    if raw:
        if dataset_exists(raw=raw):
            dataset = Dataset.load_from_disk(str(path))
            print(dataset.cleanup_cache_files())
            return dataset
        else:
            return download_raw_dataset()

    if context_length:
        if dataset_exists(
            split_by_context_length=split_by_context_length,
            preprocessed=preprocessed,
            preprocessed_name=preprocessed_name,
            context_length=context_length
        ):
            dataset = Dataset.load_from_disk(str(path))
            print(dataset.cleanup_cache_files())
            return dataset
    else:
        datasets = {}
        for context_length_dir in path.iterdir():
            context_length = context_length_dir.stem
            if context_length.isdigit():
                context_length = int(context_length)
                if dataset_exists(
                    split_by_context_length=split_by_context_length,
                    preprocessed=preprocessed,
                    preprocessed_name=preprocessed_name,
                    context_length=context_length
                ):
                    dataset = Dataset.load_from_disk(str(context_length_dir))
                    print(dataset.cleanup_cache_files())
                    datasets[context_length] = dataset
        if len(datasets) > 0:
            return DatasetDict(datasets)
        else:
            raise ValueError("No datasets found")


def download_raw_dataset():
    path = get_dataset_path(raw=True)
    path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]
    dataset.save_to_disk(str(path))
    return dataset


def dataset_exists(
    raw: bool = False,
    split_by_context_length: bool = False,
    preprocessed: bool = False,
    preprocessed_name: str = None,
    context_length: int = None
):
    path = get_dataset_path(
            raw=raw,
            split_by_context_length=split_by_context_length,
            preprocessed=preprocessed,
            preprocessed_name=preprocessed_name,
            context_length=context_length
    )
    if path.exists():
        dataset = Dataset.load_from_disk(str(path))
        if len(dataset) > 0:
            return True
    return False


def save_dataset(
    dataset: Dataset,
    raw: bool = False,
    split_by_context_length: bool = False,
    preprocessed: bool = False,
    preprocessed_name: str = None,
    context_length: int = None,
    overwrite: bool = False

):
    path = get_dataset_path(
        raw=raw,
        split_by_context_length=split_by_context_length,
        preprocessed=preprocessed,
        preprocessed_name=preprocessed_name,
        context_length=context_length
    )
    if path.exists():
        if overwrite:
            try:
                shutil.rmtree(path)
                dataset.save_to_disk(str(path))
                print(f"Overwritten dataset at {path}")
            except:
                print(f"Failed to delete folder at {path}. Could not save dataset.")
    else:
        dataset.save_to_disk(str(path))
