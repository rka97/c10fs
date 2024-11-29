import time
import copy
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import model
import argparse

class TrainingConfig:
    def __init__(self):
        # Training parameters
        self.epochs = 20
        self.batch_size = 512
        self.momentum = 0.9
        self.weight_decay = 0.256
        self.weight_decay_bias = 0.004
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type != "cpu" else torch.float32

        # EMA parameters
        self.ema_update_freq = 5
        self.ema_rho = 0.99 ** self.ema_update_freq

        # Learning rate schedule
        self.lr_max = 2e-3
        self.lr_final = 2e-4
        self.warmup_steps = 194
        self.decay_steps = 582

        # Data directory
        self.data_dir = "/tmp/datasets/"

def get_lr_schedule(config):
    return torch.cat([
        torch.linspace(0, config.lr_max, config.warmup_steps),
        torch.linspace(config.lr_max, config.lr_final, config.decay_steps),
    ])

def update_ema(train_model, valid_model, rho):
    train_weights = train_model.state_dict().values()
    valid_weights = valid_model.state_dict().values()
    for train_weight, valid_weight in zip(train_weights, valid_weights):
        if valid_weight.dtype in [torch.float16, torch.float32]:
            valid_weight *= rho
            valid_weight += (1 - rho) * train_weight

def preprocess_data(data, config):
    data = torch.tensor(data, device=config.device).to(config.dtype)
    mean = torch.tensor([125.31, 122.95, 113.87], device=config.device).to(config.dtype)
    std = torch.tensor([62.99, 62.09, 66.70], device=config.device).to(config.dtype)
    data = (data - mean) / std
    data = data.permute(0, 3, 1, 2) # Permute data from NHWC to NCHW format
    return data

def prepare_data(config):
    # Load and preprocess data
    train = torchvision.datasets.CIFAR10(root=config.data_dir, download=True)
    valid = torchvision.datasets.CIFAR10(root=config.data_dir, train=False)

    train_data = preprocess_data(train.data, config)
    valid_data = preprocess_data(valid.data, config)

    train_targets = torch.tensor(train.targets, device=config.device)
    valid_targets = torch.tensor(valid.targets, device=config.device)

    # Pad training data
    train_data = nn.ReflectionPad2d(4)(train_data)

    # Create datasets
    train_dataset = TensorDataset(train_data, train_targets)
    valid_dataset = TensorDataset(valid_data, valid_targets)

    return train_dataset, valid_dataset

def random_crop(data, crop_size=(32, 32)):
    crop_h, crop_w = crop_size
    h = data.size(2)
    w = data.size(3)
    x = torch.randint(w - crop_w, size=(1,))[0]
    y = torch.randint(h - crop_h, size=(1,))[0]
    return data[:, :, y : y + crop_h, x : x + crop_w]

class CifarTransform:
    def __init__(self, crop_size=(32, 32)):
        self.crop_size = crop_size

    def __call__(self, batch):
        data, targets = batch
        # Random crop
        data = random_crop(data, self.crop_size)
        # Random horizontal flip of first half
        data[:len(data)//2] = torch.flip(data[:len(data)//2], [-1])
        return data, targets

def train(seed=0, config=None):
    config = TrainingConfig() if config is None else config

    # Set random seed
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Load and prepare data
    train_dataset, valid_dataset = prepare_data(config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Print hardware info on first run
    if seed == 0:
        if config.device.type == "cuda":
            print("Device :", torch.cuda.get_device_name(config.device.index))
        print("Dtype  :", config.dtype)
        print()

    start_time = time.perf_counter()

    # Initialize model
    weights = model.patch_whitening(train_dataset.tensors[0][:10000, :, 4:-4, 4:-4])
    train_model = model.Model(weights, c_in=3, c_out=10, scale_out=0.125)
    train_model.to(config.dtype)

    # Handle BatchNorm precision
    for module in train_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    train_model.to(config.device)

    # Setup optimizers
    weights_params = [p for p in train_model.parameters() if p.requires_grad and len(p.shape) > 1]
    bias_params = [p for p in train_model.parameters() if p.requires_grad and len(p.shape) <= 1]

    lr_schedule = get_lr_schedule(config)
    lr_schedule_bias = 64.0 * lr_schedule

    optimizer_weights = optim.SGD(
        weights_params,
        lr=lr_schedule[0],
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True
    )

    optimizer_bias = optim.SGD(
        bias_params,
        lr=lr_schedule_bias[0],
        momentum=config.momentum,
        weight_decay=config.weight_decay_bias,
        nesterov=True
    )

    valid_model = copy.deepcopy(train_model)
    transform = CifarTransform()

    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")
    print("\nepoch    batch    train time [sec]    validation accuracy")

    train_time = 0.0
    batch_count = 0

    for epoch in range(1, config.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        train_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Apply transformations
            data, target = transform((data, target))
            batch_count += 1

            # Update learning rates
            lr_index = min(batch_count, len(lr_schedule) - 1)
            for param_group in optimizer_weights.param_groups:
                param_group['lr'] = lr_schedule[lr_index]
            for param_group in optimizer_bias.param_groups:
                param_group['lr'] = lr_schedule_bias[lr_index]

            optimizer_weights.zero_grad()
            optimizer_bias.zero_grad()

            logits = train_model(data)
            loss = model.label_smoothing_loss(logits, target, alpha=0.2)
            loss.sum().backward()

            optimizer_weights.step()
            optimizer_bias.step()

            if batch_idx % config.ema_update_freq == 0:
                update_ema(train_model, valid_model, config.ema_rho)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_time += time.perf_counter() - start_time

        # Validation
        valid_correct = []
        valid_model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(config.device), target.to(config.device)

                # Test-time augmentation
                logits1 = valid_model(data)
                logits2 = valid_model(torch.flip(data, [-1]))
                logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)

                correct = logits.max(dim=1)[1] == target
                valid_correct.append(correct.type(torch.float64))

        valid_acc = torch.mean(torch.cat(valid_correct)).item()
        print(f"{epoch:5} {batch_count:8d} {train_time:19.2f} {valid_acc:22.4f}")

    return valid_acc

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with configurable data directory')
    parser.add_argument('--data-dir',
                      default='/run/media/robo/Data/datasets',
                      help='Directory for dataset storage')
    return parser.parse_args()

def main():
    args = parse_args()
    config = TrainingConfig()
    config.data_dir = args.data_dir
    accuracies = []
    threshold = 0.94
    for run in range(100):
        valid_acc = train(seed=run, config=config)
        accuracies.append(valid_acc)

        # Print accumulated results
        within_threshold = sum(acc >= threshold for acc in accuracies)
        acc = threshold * 100.0
        print()
        print(f"{within_threshold} of {run + 1} runs >= {acc} % accuracy")
        mean = sum(accuracies) / len(accuracies)
        variance = sum((acc - mean)**2 for acc in accuracies) / len(accuracies)
        std = variance**0.5
        print(f"Min  accuracy: {min(accuracies)}")
        print(f"Max  accuracy: {max(accuracies)}")
        print(f"Mean accuracy: {mean} +- {std}")
        print()

if __name__ == "__main__":
    main()
