from setuptools import setup

setup(
    name="bert",
    version="0.1",
    description="Train BERT models.",
    author="Christian Wallenwein",
    packages=["model", "data"],
    install_requires=[
        "torch",
        "numpy",
        "datasets",
        "tqdm",
        "nltk",
        "transformers",
        "torchmetrics",
        "wandb",
    ],
    # scripts=["scripts/prepare_dataset.py", "scripts/train_model.py"],
)
