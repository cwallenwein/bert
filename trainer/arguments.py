from dataclasses import dataclass


@dataclass
class TrainingArguments:
    batch_size: int
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    save_model_after_training: bool = True

    @classmethod
    def from_dict(cls, arguments: dict):
        return TrainingArguments(
            batch_size=arguments["batch_size"],
            learning_rate=arguments["learning_rate"],
            beta1=arguments["beta1"],
            beta2=arguments["beta2"],
            save_model_after_training=arguments["save_model_after_training"]
        )
