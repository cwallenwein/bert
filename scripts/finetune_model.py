import argparse

import torch
from trainer.arguments import TrainingArguments
from trainer.finetuning import TrainerForSequenceClassificationFinetuning
from model.bert import BertConfig, BertModelForPretraining, BertModelForSequenceClassification, BertModel
from datasets import load_dataset
import transformers
from transformers import BertTokenizer


def finetune(
    wandb_run_name: str,
    learning_rate: float = 1e-4,
    epochs: int = 5,
    scheduler: str = "CosineAnnealingLR",
    p_dropout: float = 0.1
):
    pretrained_bert = load_pretrained_model(wandb_run_name)
    model = BertModelForSequenceClassification(
        pretrained_model=pretrained_bert,
        num_classes=3,
        learning_rate=learning_rate,
        scheduler=scheduler,
        p_dropout=p_dropout,

    )
    tokenizer = load_tokenizer()
    mnli = load_mnli(tokenizer)

    training_args = TrainingArguments(
        micro_batch_size=16,
        macro_batch_size=16,
        device="cuda",
        use_torch_compile=True,
        model_dtype="bfloat16"
    )

    finetuning_trainer = TrainerForSequenceClassificationFinetuning(training_args, verbose=False)

    finetuning_trainer.train(
        model=model,
        training_dataset=mnli["train"],
        validation_dataset=mnli["validation_matched"],
        epochs=epochs
    )


def load_pretrained_model(wandb_run_name: str) -> BertModel:
    config = BertConfig(
        d_model=768,
        n_layers=12,
        context_length=128,
        n_heads=12,
        feed_forward_intermediate_size=3072,
        attention_implementation="pytorch",
        p_embedding_dropout=0.1,
        p_attention_dropout=0.1,
        p_feed_forward_dropout=0.1,
    )

    model = BertModelForPretraining(config)

    if torch.cuda.is_available():
        checkpoint = torch.load(f"experiments/{wandb_run_name}/checkpoint.pt", weights_only=True)
    else:
        checkpoint = torch.load(
            f"experiments/{wandb_run_name}/checkpoint.pt",map_location=torch.device("cpu"), weights_only=True
        )

    model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(model_state_dict)
    return model.bert


def load_mnli(tokenizer):
    mnli = load_dataset("glue", "mnli")

    transformers.logging.set_verbosity_error()

    def tokenize(sample):
        return tokenizer(
            sample["premise"], sample["hypothesis"],
            max_length=128,
            padding="max_length",
            truncation=True
        )

    mnli_tokenized = mnli.map(
        tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=[
            "premise",
            "hypothesis",
            "idx"
        ]
    )
    mnli_tokenized = mnli_tokenized.rename_column("label", "labels")
    return mnli_tokenized


def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


def main():
    # Define parser
    parser = argparse.ArgumentParser(description="Finetune BERT model")

    # Finetuning arguments
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--p_dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()
    finetune(**args.__dict__)


if __name__ == "__main__":
    main()
