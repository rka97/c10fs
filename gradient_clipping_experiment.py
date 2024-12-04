import torch
import matplotlib.pyplot as plt
from train import train, TrainConfig
import numpy as np
from typing import Dict, List, Tuple

def run_experiment():
    # Define experiment configurations
    experiments = [
        ("SepClip+NoWarmup", TrainConfig(seed=42, max_grad_norm=20.0, use_lr_warmup=False, separate_clipping=True)),
        ("GlobalClip+NoWarmup", TrainConfig(seed=42, max_grad_norm=20.0, use_lr_warmup=False, separate_clipping=False)),
        ("SepClip+Warmup", TrainConfig(seed=42, max_grad_norm=20.0, use_lr_warmup=True, separate_clipping=True)),
        ("GlobalClip+Warmup", TrainConfig(seed=42, max_grad_norm=20.0, use_lr_warmup=True, separate_clipping=False)),
        ("NoClip+Warmup", TrainConfig(seed=42, max_grad_norm=None, use_lr_warmup=True)),
        ("NoClip+NoWarmup", TrainConfig(seed=42, max_grad_norm=None, use_lr_warmup=False))
    ]
    
    results: Dict[str, Tuple[List[float], List[float]]] = {}
    
    # Run all experiments
    for name, config in experiments:
        print(f"\nRunning experiment: {name}")
        print(f"Grad clipping: {config.max_grad_norm}, LR warmup: {config.use_lr_warmup}")
        val_accuracies, train_losses = train(config)
        results[name] = (val_accuracies, train_losses)
    
    # Create subplots for accuracy and loss
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot validation accuracies
    for name, (accuracies, _) in results.items():
        epochs = range(1, len(accuracies) + 1)
        ax1.plot(epochs, accuracies, label=name, marker='o')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Validation Accuracy vs Epoch')
    ax1.grid(True)
    ax1.legend()
    
    # Plot training losses
    for name, (_, losses) in results.items():
        epochs = range(1, len(losses) + 1)
        ax2.plot(epochs, losses, label=name, marker='o')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss vs Epoch')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_comparison.png')
    plt.close()

    print("\nFinal Results:")
    for name, (accuracies, _) in results.items():
        print(f"{name}: {accuracies[-1]:.4f}")
    print("\nPlot saved as training_comparison.png")

if __name__ == "__main__":
    run_experiment()
