import argparse
import plotter
import torch
from trainer import Trainer

import os
import json
from datetime import datetime


class TrainingConfig:
    def __init__(self):
        # Training parameters
        self.epochs = 2
        self.batch_size = 512
        self.momentum = 0.9
        self.weight_decay = 0.256
        self.weight_decay_bias = 0.004
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.test_model = False

        # EMA parameters
        self.ema_update_freq = 5
        self.ema_rho = 0.99**self.ema_update_freq

        # Local SGD parameters
        self.num_clients = 4
        self.local_steps = 50

        # Learning rate parameters
        self.lr_max = 2e-3
        self.lr_final = 2e-4
        self.warmup_steps = 194
        self.decay_steps = 582

        # Data directory
        self.data_dir = "/tmp/datasets/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script with configurable data directory"
    )
    parser.add_argument(
        "--data-dir", default="/tmp/datasets", help="Directory for dataset storage"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test-model", action="store_true", help="Use a smaller test model"
    )

    return parser.parse_args()

import os
import json
from datetime import datetime

def main():
    args = parse_args()
    config = TrainingConfig()
    config.data_dir = args.data_dir
    config.test_model = args.test_model

    trainer = Trainer(config)
    experiment_specs_dir = "experiment_specs"
    experiment_results_dir = "experiment_results"
    os.makedirs(experiment_results_dir, exist_ok=True)

    for experiment_file in os.listdir(experiment_specs_dir):
        if experiment_file.endswith(".json"):
            with open(os.path.join(experiment_specs_dir, experiment_file), "r") as f:
                experiment_spec = json.load(f)

            print(f"\nTraining with experiment: {experiment_file}")
            print("=" * 50)
            results = trainer.train(
                experiment_spec["optimizer"],
                seed=args.seed,
                num_local_steps=experiment_spec["num_local_steps"],
                outer_optimizer=experiment_spec.get("outer_optimizer", "sgd"),
                outer_lr=experiment_spec.get("outer_lr", 1.0),
            )
            results["metadata"] = experiment_spec

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file_name = f"{timestamp}_{experiment_file}"
            with open(os.path.join(experiment_results_dir, result_file_name), "w") as f:
                json.dump(results, f)

            print(
                f"\nFinal validation accuracy ({experiment_spec['optimizer']}): {results['final_acc']:.4f}"
            )
            print(
                f"\nResults have been saved to {result_file_name}. Use plotter to generate plots."
            )


if __name__ == "__main__":
    main()
