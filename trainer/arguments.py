import torch
from dataclasses import dataclass, fields


@dataclass
class TrainingArguments:
    micro_batch_size: int
    macro_batch_size: int
    device: str = "mps"
    save_model_after_training: bool = True
    use_torch_compile: bool = True
    use_gradient_clipping: bool = True
    gradient_clipping_value: float = 0.5
    model_dtype: str = "float32"

    def __post_init__(self):
        assert self.micro_batch_size % 8 == 0
        assert self.macro_batch_size % self.micro_batch_size == 0
        self.model_dtype: torch.dtype = parse_dtype(self.model_dtype)
        self.gradient_accumulation_steps = self.macro_batch_size // self.micro_batch_size

    @classmethod
    def from_dict(cls, arguments: dict):
        field_names = {field.name for field in fields(cls)}
        return TrainingArguments(**{key: value for key, value in arguments.items() if key in field_names})


def parse_dtype(dtype: str) -> torch.dtype:
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    try:
        return dtype_mapping[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}")
