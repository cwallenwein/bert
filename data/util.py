from nltk import tokenize
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BertTokenizerFast


def split_into_sentences(text: str):
    return tokenize.sent_tokenize(text)


def append_to_batch(batch: dict, sample: dict):
    assert batch.keys() == sample.keys()

    for key in batch.keys():
        batch[key].append(sample[key])
    return batch


def get_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = 2 ** 18
    return tokenizer


def get_dataset_path(
    raw: bool = False,
    split_by_context_length: bool = False,
    mlm_and_nsp: bool = False,
    context_length: int = None,
) -> Path:
    if raw:
        assert not context_length and not split_by_context_length and not mlm_and_nsp
    if split_by_context_length:
        assert context_length and not raw and not mlm_and_nsp
    if mlm_and_nsp:
        assert context_length and not raw and not split_by_context_length

    tokenizer = get_tokenizer()
    tokenizer_name = tokenizer.name_or_path.replace("/", "-")
    dataset_name = "wikipedia"

    base_path = Path(__file__).parent / "datasets"
    if raw:
        dataset_path = base_path / dataset_name / "raw"
    elif split_by_context_length:
        dataset_path = base_path / dataset_name / "split_by_context_length" / tokenizer_name
    elif mlm_and_nsp:
        dataset_path = base_path / dataset_name / "mlm_and_nsp" / tokenizer_name
    else:
        raise ValueError("Invalid arguments")

    if context_length:
        dataset_path = dataset_path / str(context_length)

    return dataset_path


def get_dataset(
    raw: bool = False,
    split_by_context_length: bool = False,
    mlm_and_nsp: bool = False,
    context_length: int = None
) -> Dataset | DatasetDict:
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
        return load_dataset_from_disk(path)
    else:
        datasets = {}
        for context_length_dir in path.iterdir():
            context_length = context_length_dir.stem
            if context_length.isdigit():
                context_length = int(context_length)
                datasets[context_length] = load_dataset_from_disk(context_length_dir)
        if len(datasets) > 0:
            return DatasetDict(datasets)
        else:
            raise ValueError("No datasets found")


def load_dataset_from_disk(path: Path) -> Dataset:
    if path.exists():
        dataset = Dataset.load_from_disk(str(path))
        if len(dataset) > 0:
            return dataset
        else:
            raise ValueError(f"Dataset at path {path} is empty")
    else:
        raise FileNotFoundError(f"Dataset at path {path} not found")


def save_dataset(
    dataset: Dataset,
    raw: bool = False,
    split_by_context_length: bool = False,
    mlm_and_nsp: bool = False,
    context_length: int = None
):
    path = get_dataset_path(
        raw=raw,
        split_by_context_length=split_by_context_length,
        mlm_and_nsp=mlm_and_nsp,
        context_length=context_length
    )
    dataset.save_to_disk(str(path))
