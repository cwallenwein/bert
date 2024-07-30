from dataclasses import dataclass, fields


@dataclass
class TrainingArguments:
    batch_size: int
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    device: str = "mps"
    save_model_after_training: bool = True
    gradient_accumulation_steps: int = 1

    @classmethod
    def from_dict(cls, arguments: dict):
        field_names = {field.name for field in fields(cls)}
        return TrainingArguments(**{key: value for key, value in arguments.items() if key in field_names})