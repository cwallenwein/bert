import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BertConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    context_length: int = 128
    vocab_size: int = 30522
    n_segments: int = 2
    initializer_range: float = 0.02
    attention_dropout: float = 0.1
    feed_forward_dropout: float = 0.1
    feed_forward_activation: str = "gelu"
    feed_forward_intermediate_size: int = 512

    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, config: dict, pretrained_tensorflow: bool = False):
        if pretrained_tensorflow:
            return BertConfig(
                d_model=config["hidden_size"],
                n_layers=config["num_hidden_layers"],
                n_heads=config["num_attention_heads"],
                context_length=config["max_position_embeddings"],
                vocab_size=config["vocab_size"],
                n_segments=config["type_vocab_size"],
                initializer_range=config["initializer_range"],
                attention_dropout=config["attention_probs_dropout_prob"],
                feed_forward_dropout=config["hidden_dropout_prob"],
                feed_forward_activation=config["hidden_act"],
                feed_forward_intermediate_size=config["intermediate_size"],
            )
        else:
            return BertConfig(
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                context_length=config["context_length"],
                vocab_size=config["vocab_size"],
                n_segments=config["n_segments"],
                initializer_range=config["initializer_range"],
                attention_dropout=config["attention_dropout"],
                feed_forward_dropout=config["feed_forward_dropout"],
                feed_forward_activation=config["feed_forward_activation"],
                feed_forward_intermediate_size=config["feed_forward_intermediate_size"],
            )

    @classmethod
    def from_json(cls, filename, pretrained_tensorflow: bool = False):
        path = Path(filename)
        with open(path, "r") as f:
            config = json.load(f)
            return BertConfig.from_dict(config, pretrained_tensorflow=pretrained_tensorflow)
