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
        self.optimizer = "sgd"  # Base optimizer for clients
        self.base_batch_size = 512  # This will be divided by num_clients
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
        self.outer_optimizer = "sgd"
        self.outer_lr = 1.0

        # Learning rate parameters
        self.lr_max = 2e-3
        self.lr_final = 2e-4
        self.warmup_steps = 194
        self.decay_steps = 582

        # Data directory
        self.data_dir = "/tmp/datasets/"
        
        # Set batch size based on num_clients
        self.batch_size = self.base_batch_size // self.num_clients

    def update_from_spec(self, spec_file):
        """Update config parameters from a specification file"""
        with open(spec_file, 'r') as f:
            spec = json.load(f)
            
        # Update parameters from spec
        if 'outer_optimizer' in spec:
            self.outer_optimizer = spec['outer_optimizer']
        if 'outer_lr' in spec:
            self.outer_lr = spec['outer_lr']
        if 'num_clients' in spec:
            self.num_clients = spec['num_clients']
        if 'num_local_steps' in spec:
            self.local_steps = spec['num_local_steps']
            
        # Always update batch_size based on num_clients
        self.batch_size = self.base_batch_size // self.num_clients


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
            experiment_path = os.path.join(experiment_specs_dir, experiment_file)
            
            # Update config from spec file
            config.update_from_spec(experiment_path)
            
            print(f"\nTraining with experiment: {experiment_file}")
            print("=" * 50)
            print(f"Batch size per client: {config.batch_size}")
            print(f"Number of clients: {config.num_clients}")
            print(f"Local steps: {config.local_steps}")
            print(f"Outer optimizer: {config.outer_optimizer}")
            print(f"Outer learning rate: {config.outer_lr}")
            
            # Create new trainer instance for each experiment
            trainer = Trainer(config)
            results = trainer.train(seed=args.seed)
            
            # Load spec for metadata
            with open(experiment_path, "r") as f:
                experiment_spec = json.load(f)
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
