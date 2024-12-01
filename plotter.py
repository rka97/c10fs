import matplotlib.pyplot as plt

import json
import matplotlib.pyplot as plt

def plot_results(results_file):
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
    plt.close()
