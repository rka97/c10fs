import json
import matplotlib.pyplot as plt
import argparse

import os

def plot_results_from_file(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    plt.figure(figsize=(15, 5))

    # Plot training losses
    plt.subplot(1, 3, 1)
    plt.plot(results['train_losses'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracies
    plt.subplot(1, 3, 2)
    plt.plot(results['valid_accs'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)

    # Plot learning rates
    plt.subplot(1, 3, 3)
    plt.plot(results['learning_rates'])
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(results_file.replace('.json', '.png'))
    plt.show()
    plt.close()


def plot_results(results_dir):
    results = []
    for result_file in os.listdir(results_dir):
        if result_file.endswith('.json'):
            with open(os.path.join(results_dir, result_file), 'r') as f:
                results.append(json.load(f))

    plt.figure(figsize=(15, 5))

    for result in results:
        plt.subplot(1, 3, 1)
        plt.plot(result['train_losses'], label=f"{result['metadata']['optimizer']} - {result['metadata']['num_local_steps']} steps")
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss vs Epoch')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(result['valid_accs'], label=f"{result['metadata']['optimizer']} - {result['metadata']['num_local_steps']} steps")
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs Epoch')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(result['learning_rates'], label=f"{result['metadata']['optimizer']} - {result['metadata']['num_local_steps']} steps")
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'combined_results.png'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('results_path', help='Path to results JSON file or directory')
    args = parser.parse_args()
    results_path = args.results_path
    if not os.path.exists(results_path):
        raise Exception(f"{results_path} does not exist.")
    elif os.path.isdir(results_path):
        plot_results(results_path)
    elif os.path.isfile(results_path):
        plot_results_from_file(results_path)
    else:
        raise Exception(f"{results_path} is neither a file nor a directory.")
