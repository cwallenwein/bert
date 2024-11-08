import json
from dataclasses import dataclass, fields
from pathlib import Path

from transformers import BertConfig as HuggingfaceBertConfig


@dataclass
class BertConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    context_length: int = 128
    vocab_size: int = 30522
    n_segments: int = 2
    initializer_range: float = 0.02
    feed_forward_activation: str = "gelu"
    feed_forward_intermediate_size: int = 512
    feed_forward_bias: bool = True
    p_embedding_dropout: float = 0.1
    p_attention_dropout: float = 0.1
    p_feed_forward_dropout: float = 0.1
    attention_implementation: str = "pytorch"
    positional_information_type: str = "learned"
    attention_bias: bool = True
    layer_norm: str = "pre"
    add_final_layer_norm: bool = False
    with_next_sentence_prediction: bool = False

    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        assert self.layer_norm in ["pre", "post"]
        assert self.feed_forward_activation in ["relu", "gelu", "glu"]
        assert self.attention_implementation in [
            "from-scratch",
            "pytorch",
            "flash-attn",
        ]
        assert self.positional_information_type in [
            "learned",
            "sinusoidal",
            "scaled_sinusoidal",
        ]

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
                p_attention_dropout=config["attention_probs_dropout_prob"],
                p_feed_forward_dropout=config["hidden_dropout_prob"],
                feed_forward_activation=config["hidden_act"],
                feed_forward_intermediate_size=config["intermediate_size"],
            )
        else:
            field_names = {field.name for field in fields(cls)}
            return BertConfig(
                **{key: value for key, value in config.items() if key in field_names}
            )

    @classmethod
    def from_json(cls, filename, pretrained_tensorflow: bool = False):
        path = Path(filename)
        with open(path, "r") as f:
            config = json.load(f)
            return BertConfig.from_dict(
                config, pretrained_tensorflow=pretrained_tensorflow
            )

    @classmethod
    def from_huggingface_config(cls, config: HuggingfaceBertConfig):
        return BertConfig(
            d_model=config.hidden_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            context_length=config.max_position_embeddings,
            vocab_size=config.vocab_size,
            n_segments=config.type_vocab_size,
            initializer_range=config.initializer_range,
            p_attention_dropout=config.attention_probs_dropout_prob,
            p_feed_forward_dropout=config.hidden_dropout_prob,
            feed_forward_activation=config.hidden_act,
            feed_forward_intermediate_size=config.intermediate_size,
        )
