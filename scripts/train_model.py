import argparse
from trainer import TrainerForPreTraining
from trainer.arguments import TrainingArguments
from model import BertModel, BertConfig


def train(args):
    config = BertConfig.from_dict(args)
    model = BertModel(config)
    training_args = TrainingArguments.from_dict(args)
    trainer = TrainerForPreTraining(model, training_args)

    if args["context_length"] is not None and args["training_steps"] is not None:
        context_length = args["context_length"]
        training_steps = args["training_steps"]
        assert args["short_context_length"] is None
        assert args["long_context_length"] is None
        assert args["short_context_training_steps"] is None
        assert args["long_context_training_steps"] is None

        trainer.train(max_steps=training_steps, context_length=context_length)
    else:
        short_context_length = args["short_context_length"]
        long_context_length = args["long_context_length"]
        short_context_training_steps = args["short_context_training_steps"]
        long_context_training_steps = args["long_context_training_steps"]
        assert short_context_length is not None
        assert long_context_length is not None
        assert short_context_training_steps is not None
        assert long_context_training_steps is not None
        assert args["context_length"] is None
        assert args["training_steps"] is None

        trainer.train(max_steps=short_context_training_steps, context_length=short_context_length)
        trainer.train(max_steps=long_context_training_steps, context_length=long_context_length)


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
parser.add_argument("--context_length", type=int)

parser.add_argument("--short_context_length", type=int)
parser.add_argument("--short_context_training_steps", type=int)
parser.add_argument("--long_context_length", type=int)
parser.add_argument("--long_context_training_steps", type=int)

args = parser.parse_args()
train(args=args.__dict__)
