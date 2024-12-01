import matplotlib.pyplot as plt

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
