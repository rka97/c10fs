import argparse
import plotter
import torch
import torch.multiprocessing as mp
from trainer import Trainer
import os
import json
from datetime import datetime
from typing import List, Dict
import math
import time


class TrainingConfig:
    def __init__(self):
        # Training parameters
        self.epochs = 20
        self.optimizer = "adamw"  # Base optimizer for clients
        self.base_batch_size = 512  # This will be divided by num_clients

        self.num_processes = 2  # Number of parallel processes to run

        self.momentum = 0.9
        self.weight_decay = 0.256
        self.weight_decay_bias = 0.004
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.test_config = False
        self.compile_model = False  # Whether to compile the model with torch.compile

        # EMA parameters
        self.ema_update_freq = 5
        self.ema_rho = 0.99**self.ema_update_freq

        # Local SGD parameters
        self.num_clients = 4
        self.local_steps = 50
        self.outer_optimizer = "sgd"
        self.outer_lr = 1.0
        self.outer_momentum = 0.9  # Default momentum for Nesterov SGD

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
        with open(spec_file, "r") as f:
            spec = json.load(f)

        # Update parameters from spec
        if "outer_optimizer" in spec:
            self.outer_optimizer = spec["outer_optimizer"]
        if "outer_lr" in spec:
            self.outer_lr = spec["outer_lr"]
        if "outer_momentum" in spec:
            self.outer_momentum = spec["outer_momentum"]
        if "num_clients" in spec:
            self.num_clients = spec["num_clients"]
        if "num_local_steps" in spec:
            self.local_steps = spec["num_local_steps"]

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
        "--test-config", action="store_true", help="Use a test configuration with smaller model and reduced clients/processes"
    )
    parser.add_argument(
        "--experiment", help="Path to a specific experiment spec file to run"
    )

    return parser.parse_args()


def run_experiment(
    process_id: int,
    num_gpus: int,
    experiment_queue: List[str],
    args,
    base_config: TrainingConfig,
):
    """Run experiments assigned to a specific process"""
    # Assign GPU in round-robin fashion
    gpu_id = process_id % num_gpus if num_gpus > 0 else -1
    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)

    experiment_specs_dir = "experiment_specs"
    experiment_results_dir = "experiment_results"
    os.makedirs(experiment_results_dir, exist_ok=True)

    for experiment_file in experiment_queue:
        experiment_path = os.path.join(experiment_specs_dir, experiment_file)

        # Create a new config for this experiment
        config = TrainingConfig()
        config.data_dir = args.data_dir
        config.test_model = args.test_config
        if args.test_config:
            config.epochs = 2
            config.optimizer = "sgd"
            config.num_clients = 2
            config.num_processes = 2
        config.device = torch.device(f"cuda:{gpu_id}")

        # Update config from spec file
        config.update_from_spec(experiment_path)

        print(f"\nGPU {gpu_id} training with experiment: {experiment_file}")
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
            f"\nGPU {gpu_id} - Final validation accuracy ({experiment_spec['optimizer']}): {results['final_acc']:.4f}"
        )
        print(f"\nResults have been saved to {result_file_name}.")


def distribute_experiments(
    experiment_files: List[str], num_processes: int
) -> List[List[str]]:
    """Distribute experiments across available processes"""
    process_queues = [[] for _ in range(num_processes)]

    # Sort experiments to ensure deterministic distribution
    experiment_files.sort()

    # Distribute experiments as evenly as possible
    for i, exp_file in enumerate(experiment_files):
        process_idx = i % num_processes
        process_queues[process_idx].append(exp_file)

    return process_queues


def main():
    # Enable multiprocessing for PyTorch
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    base_config = TrainingConfig()

    # Get experiments to run
    if args.experiment:
        if not os.path.exists(args.experiment):
            raise ValueError(f"Experiment file {args.experiment} does not exist")
        experiment_files = [os.path.basename(args.experiment)]
        experiment_specs_dir = os.path.dirname(args.experiment)
        if not experiment_specs_dir:
            experiment_specs_dir = "experiment_specs"
    else:
        # Get all available experiments
        experiment_specs_dir = "experiment_specs"
        experiment_files = [
            f for f in os.listdir(experiment_specs_dir) if f.endswith(".json")
        ]

    # Get number of available GPUs and processes
    num_gpus = torch.cuda.device_count()
    num_processes = base_config.num_processes

    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
    else:
        print(f"Found {num_gpus} GPUs")

    print(f"Creating {num_processes} processes")
    print(f"Found {len(experiment_files)} experiments")

    # Distribute experiments across processes
    process_queues = distribute_experiments(experiment_files, num_processes)

    # Create processes
    processes = []
    for process_id in range(num_processes):
        if process_queues[
            process_id
        ]:  # Only create process if there are experiments to run
            p = mp.Process(
                target=run_experiment,
                args=(
                    process_id,
                    num_gpus,
                    process_queues[process_id],
                    args,
                    base_config,
                ),
            )
            p.start()
            processes.append(p)
        time.sleep(5)

    # Wait for all processes to complete
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
