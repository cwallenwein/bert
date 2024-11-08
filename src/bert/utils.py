import os
from pathlib import Path


def get_repository_root_dir() -> Path:
    repository_root_dir = Path(__file__).parent.parent.parent

    return repository_root_dir


def get_tokenizers_base_dir() -> Path:
    """
    Get the absolute path to the tokenizers directory based on the TOKENIZERS_BASE_DIR
    environment variable.

    Returns:
        Path: Absolute path to the tokenizers directory
    """
    tokenizers_base_dir = os.getenv("TOKENIZERS_BASE_DIR")
    assert tokenizers_base_dir is not None, "TOKENIZERS_BASE_DIR must be set"
    assert not os.path.isabs(
        tokenizers_base_dir
    ), "TOKENIZERS_BASE_DIR must be relative path from repo root"

    repository_root_dir = get_repository_root_dir()

    return Path(repository_root_dir / tokenizers_base_dir)


def get_datasets_base_dir() -> Path:
    """
    Get the absolute path to the datasets directory based on the DATASETS_BASE_DIR
    environment variable.

    Returns:
        Path: Absolute path to the tokenizers directory
    """
    datasets_base_dir = os.getenv("DATASETS_BASE_DIR")
    assert datasets_base_dir is not None, "DATASETS_BASE_DIR must be set"
    assert not os.path.isabs(
        datasets_base_dir
    ), "DATASETS_BASE_DIR must be relative path from repo root"

    repository_root_dir = get_repository_root_dir()

    return Path(repository_root_dir / datasets_base_dir)


def get_datasets_raw_dir() -> Path:
    """
    Get the absolute path to the datasets directory based on the DATASETS_RAW_DIR
    environment variable.

    Returns:
        Path: Absolute path to the tokenizers directory
    """
    datasets_raw_dir = os.getenv("DATASETS_RAW_DIR")
    assert datasets_raw_dir is not None, "DATASETS_RAW_DIR must be set"
    assert not os.path.isabs(
        datasets_raw_dir
    ), "DATASETS_RAW_DIR must be relative path from repo root"

    repository_root_dir = get_repository_root_dir()

    return Path(repository_root_dir / datasets_raw_dir)


def get_datasets_processed_dir() -> Path:
    """
    Get the absolute path to the datasets directory based on the DATASETS_PROCESSED_DIR
    environment variable.

    Returns:
        Path: Absolute path to the tokenizers directory
    """
    datasets_processed_dir = os.getenv("DATASETS_PROCESSED_DIR")
    assert datasets_processed_dir is not None, "DATASETS_PROCESSED_DIR must be set"
    assert not os.path.isabs(
        datasets_processed_dir
    ), "DATASETS_PROCESSED_DIR must be relative path from repo root"

    repository_root_dir = get_repository_root_dir()

    return Path(repository_root_dir / datasets_processed_dir)


def get_datasets_cache_dir() -> Path:
    """
    Get the absolute path to the datasets directory based on the DATASETS_PROCESSED_DIR
    environment variable.

    Returns:
        Path: Absolute path to the tokenizers directory
    """
    datasets_cache_dir = os.getenv("DATASETS_CACHE_DIR")
    assert datasets_cache_dir is not None, "DATASETS_CACHE_DIR must be set"
    assert not os.path.isabs(
        datasets_cache_dir
    ), "DATASETS_CACHE_DIR must be relative path from repo root"

    repository_root_dir = get_repository_root_dir()

    return Path(repository_root_dir / datasets_cache_dir)
