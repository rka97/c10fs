import json
import matplotlib.pyplot as plt
import argparse
import os
from tabulate import tabulate

def plot_results_from_file(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    plt.figure(figsize=(10, 5))

    # Plot training losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(results['train_losses']) + 1), results['train_losses'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(results['valid_accs']) + 1), results['valid_accs'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.savefig(results_file.replace('.json', '.png'))
    plt.show()
    plt.close()


def plot_results(results_dir):
    results = []
    table_data = []
    
    # Dictionary to track best results for each (optimizer, local_steps) combination
    best_results = {}
    
    for result_file in os.listdir(results_dir):
        if result_file.endswith('.json'):
            with open(os.path.join(results_dir, result_file), 'r') as f:
                result = json.load(f)
                results.append(result)
                
                # Extract data for table
                metadata = result['metadata']
                outer_opt = metadata.get('outer_optimizer', 'N/A')
                local_steps = metadata.get('num_local_steps', 'N/A')
                outer_lr = metadata.get('outer_lr', 'N/A')
                final_acc = result['final_acc']
                
                table_data.append([
                    f"{outer_lr:.3f}",
                    outer_opt,
                    local_steps,
                    f"{final_acc:.4f}"
                ])
                
                # Track best results
                key = (outer_opt, local_steps)
                if key not in best_results or final_acc > best_results[key]['acc']:
                    best_results[key] = {
                        'acc': final_acc,
                        'lr': outer_lr,
                        'result': result
                    }
    
    # Print all results table
    headers = ['Outer LR', 'Outer Optimizer', 'Local Steps', 'Final Val Acc']
    print("\nAll Experiment Results:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("\n")
    
    # Print best results table
    best_table_data = []
    for (opt, steps), data in best_results.items():
        best_table_data.append([
            f"{data['lr']:.3f}",
            opt,
            steps,
            f"{data['acc']:.4f}"
        ])
    
    print("Best Results per (Optimizer, Local Steps):")
    print(tabulate(best_table_data, headers=headers, tablefmt='grid'))
    print("\n")

    # Plot all results
    plt.figure(figsize=(10, 5))
    for result in results:
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(result['train_losses']) + 1), result['train_losses'], 
                label=f"{result['metadata']['outer_optimizer']} - {result['metadata']['num_local_steps']} steps (lr={result['metadata']['outer_lr']:.3f})")
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('All Results: Training Loss vs Epoch')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(result['valid_accs']) + 1), result['valid_accs'], 
                label=f"{result['metadata']['outer_optimizer']} - {result['metadata']['num_local_steps']} steps (lr={result['metadata']['outer_lr']:.3f})")
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('All Results: Validation Accuracy vs Epoch')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)


    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'all_results.png'))
    plt.show()
    plt.close()

    # Plot best results
    plt.figure(figsize=(10, 5))
    for (opt, steps), data in best_results.items():
        result = data['result']
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(result['train_losses']) + 1), result['train_losses'], 
                label=f"{opt} - {steps} local steps")
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Best Results: Training Loss vs Epoch')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(result['valid_accs']) + 1), result['valid_accs'], 
                label=f"{opt} - {steps} local steps")
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Best Results: Validation Accuracy vs Epoch')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)


    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'best_results.png'))
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
