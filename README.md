# BERT PyTorch Implementation (work in progress)

## Introduction
This is an implementation of the BERT architecture (Bidirectional Encoder Representations from Transformers [paper](https://arxiv.org/abs/1810.04805)).
This repository provides a PyTorch implementation of BERT from scratch, including the model architecture, data preprocessing and training.


## Installation

### Setup environment
```
conda env create -f ./env/env.yaml
conda activate bert
python -m ipykernel install --user --name=bert
```


### Remove environment
```
conda deactivate
conda remove -n bert --all -y
jupyter kernelspec uninstall bert -y
```

## Usage
`python scripts/prepare_dataset.py --num_samples=10000`  

`python scripts/train_model.py --d_model=64 --n_layers=4 --n_heads=4 --batch_size=8 --training_steps=1000`

## Implementation

### Datasets
Unlike in the [original BERT paper](https://arxiv.org/abs/1810.04805), this implementation doesn't yet use the bookcorpus dataset for the pretraining.
In the [Huggingface version of the bookcorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) most samples are short, independent sentences from books without any connection to other sentences.

### Preprocessing
The preprocessing is split into three distinct parts:
1. Splitting by context length: Each sample from original dataset is split into multiple samples of length _context_length_. Each sample must contain at least two sentences.
2. Preparing next sentence prediction (NSP): For NSP, half of the samples will contain two consecutive sequences of sentences (_IsNext_) and the other half will contain two non-consecutive sequences of sentences (_NotNext_). For the _IsNext_-case a sample will randomly be split into two segments at a sentence delimiter. For the _NotNext_-case two samples will be randomly split into two segments each that will be combined with a segment form the other sample. 
3. Preparing masked language modeling (MLM): For MLM, (by default) 15% of tokens will be masked. Of these, 80% will be replaced by the [MASK] token, 10% will be replaced by a random token and the remaining 10% will not be replaced.