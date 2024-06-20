import argparse
from trainer import SimpleTrainer
from trainer.arguments import TrainingArguments
from data.load import get_dataset
from model import BertModel, BertConfig


def train(args):
    config = BertConfig.from_dict(args)
    model = BertModel(config)
    training_args = TrainingArguments.from_dict(args)
    trainer = SimpleTrainer(model, training_args)

    training_steps = args["training_steps"]
    long_context_length = args["long_context_length"]
    long_context_pct = args["long_context_pct"]

    if long_context_length is None:
        assert long_context_pct == 0
        dataset = get_dataset(mlm_and_nsp=True, context_length=args["context_length"])
        trainer.train(dataset=dataset, training_steps=training_steps)
    else:
        assert long_context_pct is not None
        dataset = get_dataset(mlm_and_nsp=True)
        assert len(dataset.keys()) == 2
        short_context_dataset_key, long_context_dataset_key = sorted(dataset.keys())
        assert args["context_length"] == int(short_context_dataset_key)
        assert long_context_length == int(long_context_dataset_key)

        short_context_dataset = dataset[short_context_dataset_key]
        long_context_dataset = dataset[long_context_dataset_key]
        trainer.train(
            dataset=short_context_dataset,
            training_steps=training_steps,
            long_context_pct=long_context_pct,
            long_context_dataset=long_context_dataset
        )


parser = argparse.ArgumentParser(description="Train BERT model")

# Model arguments
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--context_length", type=int, default=128)
parser.add_argument("--vocab_size", type=int, default=30522)
parser.add_argument("--n_segments", type=int, default=2)
parser.add_argument("--initializer_range", type=int, default=0.02)
parser.add_argument("--attention_dropout", type=float, default=0.1)
parser.add_argument("--feed_forward_dropout", type=float, default=0.1)
parser.add_argument("--feed_forward_activation", type=str, default="gelu")
parser.add_argument("--feed_forward_intermediate_size", type=int, default=512)

# Training arguments
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)


# Other arguments
parser.add_argument("--training_steps", type=int)
parser.add_argument("--long_context_length", type=int)
parser.add_argument("--long_context_pct", type=float, default=0)

args = parser.parse_args()
train(args=args.__dict__)
