import torch
import torch.nn as nn
from trainer import Trainer
from schedulefree import SGDScheduleFree
import json

class TestConfig:
    def __init__(self):
        # Minimal config for testing
        self.epochs = 2
        self.optimizer = "sgd"  # Inner optimizer
        self.batch_size = 128
        self.num_clients = 4
        self.local_steps = 1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.weight_decay_bias = 0.0001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.test_config = True
        self.data_dir = "/tmp/datasets/"
        self.compile_model = False
        
        # Schedule-Free specific settings
        self.outer_optimizer = "schedulefree"
        self.outer_lr = 1.0
        self.warmup_steps = 100

        self.lr_max = 2e-3
        self.lr_final = 2e-4
        self.warmup_steps = 194
        self.decay_steps = 582

        # EMA parameters
        self.ema_update_freq = 5
        self.ema_rho = 0.99**self.ema_update_freq


def test_schedulefree():
    config = TestConfig()
    trainer = Trainer(config)
    
    # Initialize dataset
    train_dataset = trainer.prepare_data()
    
    # Create a simple model for testing
    model = nn.Linear(10, 2).to(config.device)
    
    # Initialize Schedule-Free SGD as outer optimizer
    outer_optimizer = SGDScheduleFree(
        model.parameters(),
        lr=config.outer_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps
    )
    
    # Training loop
    outer_optimizer.train()  # Set to training mode
    for epoch in range(config.epochs):
        loss = trainer.train(seed=42)
        print(f"Epoch {epoch}, Loss: {loss}")
    
    # Save results
    results = {
        "final_loss": loss,
        "config": {
            "outer_optimizer": config.outer_optimizer,
            "outer_lr": config.outer_lr,
            "num_clients": config.num_clients,
            "local_steps": config.local_steps,
            "warmup_steps": config.warmup_steps
        }
    }
    
    with open("schedulefree_test_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    test_schedulefree()
