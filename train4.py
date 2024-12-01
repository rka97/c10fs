import argparse
import plotter
import torch
from trainer import Trainer

class TrainingConfig:
    def __init__(self):
        # Training parameters
        self.epochs = 10
        self.batch_size = 512
        self.momentum = 0.9
        self.weight_decay = 0.256
        self.weight_decay_bias = 0.004
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # EMA parameters
        self.ema_update_freq = 5
        self.ema_rho = 0.99 ** self.ema_update_freq

        # Learning rate parameters
        self.lr_max = 2e-3
        self.lr_final = 2e-4
        self.warmup_steps = 194
        self.decay_steps = 582

        # Data directory
        self.data_dir = "/tmp/datasets/"

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with configurable data directory')
    parser.add_argument('--data-dir',
                      default='/tmp/datasets',
                      help='Directory for dataset storage')
    parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    config = TrainingConfig()
    config.data_dir = args.data_dir

    trainer = Trainer(config)
    optimizers = ['sgd', 'adamw']
    results = {}

    for optimizer_name in optimizers:
        print(f"\nTraining with {optimizer_name.upper()}:")
        print("=" * 50)
        results[optimizer_name] = trainer.train(optimizer_name, seed=args.seed)
        print(f"\nFinal validation accuracy ({optimizer_name}): {results[optimizer_name]['final_acc']:.4f}")

    # Plot and save the results
    # plotter.plot_results(results) # Moved to trainer
    print(f"\nResults have been saved. Use plotter to generate plots.")

if __name__ == "__main__":
    main()
