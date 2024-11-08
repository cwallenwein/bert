import argparse

from bert.model.bert.for_mlm_pretraining import BertModelForMLM
from bert.old_trainer import TrainerForPreTraining
from bert.old_trainer.arguments import TrainingArguments
from datasets import load_from_disk


def train(args):
    model = BertModelForMLM(
        d_model=args["d_model"],
        n_layers=args["n_layers"],
        n_heads=args["n_heads"],
        vocab_size=args["vocab_size"],
        n_segments=args["n_segments"],
        initializer_range=args["initializer_range"],
        p_attention_dropout=args["p_attention_dropout"],
        p_feed_forward_dropout=args["p_feed_forward_dropout"],
        feed_forward_activation=args["feed_forward_activation"],
        feed_forward_intermediate_size=args["feed_forward_intermediate_size"],
    )
    training_args = TrainingArguments.from_dict(args)
    trainer = TrainerForPreTraining(training_args)

    dataset = load_from_disk("datasets/processed/mlm-fineweb-10BT/")

    if args["context_length"] is not None and args["training_steps"] is not None:
        # context_length = args["context_length"]
        training_steps = args["training_steps"]

        trainer.train(model, dataset, max_steps=training_steps)


parser = argparse.ArgumentParser(description="Train BERT model")

# Model arguments
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--vocab_size", type=int, default=30522)
parser.add_argument("--n_segments", type=int, default=2)
parser.add_argument("--initializer_range", type=int, default=0.02)
parser.add_argument("--p_attention_dropout", type=float, default=0.1)
parser.add_argument("--p_feed_forward_dropout", type=float, default=0.1)
parser.add_argument("--feed_forward_activation", type=str, default="gelu")
parser.add_argument("--feed_forward_intermediate_size", type=int, default=512)

# Training arguments
# parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--micro_batch_size", type=int, default=8)
parser.add_argument("--macro_batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)


# Other arguments
parser.add_argument("--training_steps", type=int)
parser.add_argument("--context_length", type=int, default=128)

args = parser.parse_args()
train(args=args.__dict__)
