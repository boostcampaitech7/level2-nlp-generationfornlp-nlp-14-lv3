import json
import os
import subprocess
import time
from pathlib import Path

SCHEDULER_FILE = os.path.join(Path(__file__).parent.parent, "scheduler.json")
ARGS_FILE = os.path.join(Path(__file__).parent.parent, "args.json")

with open(SCHEDULER_FILE, "r") as f:
    configs = json.load(f)


def update_args(new_args):
    """Update the args.json file with new arguments."""
    with open(ARGS_FILE, "r") as f:
        current_args = json.load(f)

    current_args.update(new_args)

    with open(ARGS_FILE, "w") as f:
        json.dump(current_args, f, indent=4)


def run_training():
    """Run the training script."""
    try:
        subprocess.run(["python", "-m", "src.main"], check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")


def main():
    for config in configs:
        print(f"Starting training with config: {config}")
        update_args(config)
        run_training()
        time.sleep(5)


if __name__ == "__main__":
    main()
