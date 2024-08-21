import wandb
import argparse
from functools import partial
from scripts.finetune_model import finetune


def start_finetuning_sweep(wandb_run_name: str):
    sweep_config = {
        "method": "random",
        "name": "sweep_test",
        "metric": {
            "goal": "maximize",
            "name": "val/acc"
        },
        "parameters": {
            "wandb_run_name": {"value": wandb_run_name},
            "learning_rate": {"min": 1e-5, "max": 1e-2},
            "scheduler": {"values": [
                "CosineAnnealingLR", "OneCycleLR", "DynamicWarmupStableDecayScheduler"
            ]},
            "p_dropout": {"min": 0.1, "max": 0.3}
        }
    }

    sweep_function = partial(finetune, wandb_run_name=wandb_run_name)
    sweep_id = wandb.sweep(sweep_config, project="test_sweep")
    wandb.agent(sweep_id=sweep_id, function=sweep_function, count=5)


def main():
    # Define parser
    parser = argparse.ArgumentParser(description="Sweep the finetuning of BERT")

    # Finetuning arguments
    parser.add_argument("--wandb_run_name", type=str, required=True)

    args = parser.parse_args()
    start_finetuning_sweep(**args.__dict__)


if __name__ == "__main__":
    main()
