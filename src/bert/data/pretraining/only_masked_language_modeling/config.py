import hashlib
import json
from dataclasses import dataclass


@dataclass
class MaskedLanguageModelingDatasetConfig:
    tokenizer_id_or_name: str
    raw_dataset_name: str
    num_samples: int = None
    context_length: int = 128
    p_mask: float = 0.15
    p_replacement_mask: float = 0.8
    p_replacement_random: float = 0.1

    def to_string(self):
        return json.dumps(self.__dict__, sort_keys=True)

    @classmethod
    def from_string(cls, string: str):
        return MaskedLanguageModelingDatasetConfig(**json.loads(string))

    def get_hash_value(self):
        config_as_string = self.to_string()
        hash = hashlib.md5(config_as_string.encode())
        return hash.hexdigest()[:10]
