import argparse
import matplotlib.pyplot as plt
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

def plot_results(results_dict):
    plt.figure(figsize=(15, 5))

    # Plot training losses
    plt.subplot(1, 3, 1)
    for optimizer_name, results in results_dict.items():
        plt.plot(results['train_losses'], label=f'{optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracies
    plt.subplot(1, 3, 2)
    for optimizer_name, results in results_dict.items():
        plt.plot(results['valid_accs'], label=f'{optimizer_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)

    # Plot learning rates
    plt.subplot(1, 3, 3)
    for optimizer_name, results in results_dict.items():
        plt.plot(results['learning_rates'], label=f'{optimizer_name}')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

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
    optimizers = ['adamw']
    results = {}

    for optimizer_name in optimizers:
        print(f"\nTraining with {optimizer_name.upper()}:")
        print("=" * 50)
        results[optimizer_name] = trainer.train(optimizer_name, seed=args.seed)
        print(f"\nFinal validation accuracy ({optimizer_name}): {results[optimizer_name]['final_acc']:.4f}")

    # Plot and save the results
    plot_results(results)
    print("\nResults have been plotted and saved to 'training_results.png'")

if __name__ == "__main__":
    main()
