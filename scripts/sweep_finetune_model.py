import wandb
import argparse
from functools import partial
from scripts.finetune_model import finetune


def start_finetuning_sweep(wandb_run_name: str, num_runs: int = 8):
    sweep_config = {
        "method": "random",
        "metric": {
            "goal": "maximize",
            "name": "val/accuracy"
        },
        "parameters": {
            "learning_rate": {"min": 1e-5, "max": 1e-2},
            "scheduler": {"values": [
                "CosineAnnealingLR", "OneCycleLR", "DynamicWarmupStableDecayScheduler"
            ]},
            "p_dropout": {"min": 0.1, "max": 0.3}
        }
    }

    def sweep_function():
        wandb.init()
        finetune(wandb_run_name=wandb_run_name, **wandb.config)

    sweep_id = wandb.sweep(sweep_config, project="BERT")
    wandb.agent(sweep_id=sweep_id, function=sweep_function, count=num_runs)


def main():
    # Define parser
    parser = argparse.ArgumentParser(description="Sweep the finetuning of BERT")

    # Finetuning arguments
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=8)

    args = parser.parse_args()
    start_finetuning_sweep(**args.__dict__)


if __name__ == "__main__":
    main()
