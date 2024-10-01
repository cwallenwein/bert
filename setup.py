from setuptools import find_packages, setup

setup(
    name="bert",
    version="0.1",
    author="Christian Wallenwein",
    description="Train BERT models.",
    url="https://github.com/cwallenwein/bert",
    project_urls={
        "Bug Tracker": "https://github.com/cwallenwein/bert/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "datasets",
        "tqdm",
        "nltk",
        "transformers",
        "torchmetrics",
        "wandb",
        "lightning",
        "einops",
        "progressive_scheduling@git+https://github.com/cwallenwein/progressive-scheduling.git",
    ],
    # scripts=["scripts/prepare_dataset.py", "scripts/train_model.py"],
)
