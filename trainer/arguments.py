from dataclasses import dataclass, fields


@dataclass
class TrainingArguments:
    micro_batch_size: int
    macro_batch_size: int
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    device: str = "mps"
    save_model_after_training: bool = True
    with_wandb: bool = True

    use_torch_compile: bool = True
    use_gradient_clipping: bool = False
    gradient_clipping_value: float = 0.5

    def __post_init__(self):
        assert self.micro_batch_size % 8 == 0
        assert self.macro_batch_size % self.micro_batch_size == 0
        self.gradient_accumulation_steps = self.macro_batch_size // self.micro_batch_size

    @classmethod
    def from_dict(cls, arguments: dict):
        field_names = {field.name for field in fields(cls)}
        return TrainingArguments(**{key: value for key, value in arguments.items() if key in field_names})