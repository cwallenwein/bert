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
        "lightning[pytorch-extra]",
        "datasets",
        "transformers[torch]",
        "torchmetrics",
        "einops",
        "wandb",
        "progressive_scheduling@git+https://github.com/cwallenwein/progressive-scheduling.git",
        "evaluate",
        "tqdm",
        "ipykernel",
        "python-dotenv",
        "nltk",
        "fastcore",
    ],
    extras_require={
        "dev": ["black", "flake8", "isort"],
    },
    scripts=["scripts/pretrain.py"],
)
