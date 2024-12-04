import time
import copy
import torch
import torch.nn as nn
import torchvision
import model


class TrainConfig:
    def __init__(self, seed=0, max_grad_norm=None, use_lr_warmup=True, separate_clipping=True):
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.use_lr_warmup = use_lr_warmup
        self.separate_clipping = separate_clipping
        self.epochs = 4
        self.batch_size = 512
        self.momentum = 0.9
        self.weight_decay = 0.256
        self.weight_decay_bias = 0.004
        self.ema_update_freq = 5
        self.ema_rho = 0.99 ** self.ema_update_freq
        self.base_lr = 2e-3
        self.final_lr = 2e-4

def train(config=None):
    if config is None:
        config = TrainConfig()
    
    # Lists to store metrics
    validation_accuracies = []
    training_losses = []
    
    # Configurable parameters
    epochs = config.epochs
    batch_size = config.batch_size
    momentum = config.momentum
    weight_decay = config.weight_decay
    weight_decay_bias = config.weight_decay_bias
    ema_update_freq = config.ema_update_freq
    ema_rho = config.ema_rho
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    # Create learning rate schedule
    if config.use_lr_warmup:
        # Learning rate warmup followed by decay
        lr_schedule = torch.cat([
            torch.linspace(0, config.base_lr, 194),
            torch.linspace(config.base_lr, config.final_lr, 582),
        ])
    else:
        # Constant learning rate followed by decay
        lr_schedule = torch.cat([
            torch.ones(194) * config.base_lr / 3,
            torch.linspace(config.base_lr / 3, config.final_lr, 582),
        ])

    lr_schedule_bias = 64.0 * lr_schedule

    # Print information about hardware on first run
    if config.seed == 0:
        if device.type == "cuda":
            print("Device :", torch.cuda.get_device_name(device.index))

        print("Dtype  :", dtype)
        print()

    # Start measuring time
    start_time = time.perf_counter()

    # Set random seed to increase chance of reproducability
    torch.manual_seed(config.seed)

    # Setting cudnn.benchmark to True hampers reproducability, but is faster
    torch.backends.cudnn.benchmark = True

    # Load dataset
    train_data, train_targets, valid_data, valid_targets = load_cifar10(device, dtype)

    # Compute special weights for first layer
    weights = model.patch_whitening(train_data[:10000, :, 4:-4, 4:-4])

    # Construct the neural network
    train_model = model.Model(weights, c_in=3, c_out=10, scale_out=0.125)

    # Convert model weights to half precision
    train_model.to(dtype)

    # Convert BatchNorm back to single precision for better accuracy
    for module in train_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    # Upload model to GPU
    train_model.to(device)

    # Collect weights and biases and create nesterov velocity values
    weights = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) > 1
    ]
    biases = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) <= 1
    ]

    # Copy the model for validation
    valid_model = copy.deepcopy(train_model)

    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    # Train and validate
    print("\nepoch    batch    train time [sec]    validation accuracy    avg train loss")
    train_time = 0.0
    batch_count = 0
    for epoch in range(1, epochs + 1):
        # Flush CUDA pipeline for more accurate time measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # Randomly shuffle training data
        indices = torch.randperm(len(train_data), device=device)
        data = train_data[indices]
        targets = train_targets[indices]

        # Crop random 32x32 patches from 40x40 training data
        data = [
            random_crop(data[i : i + batch_size], crop_size=(32, 32))
            for i in range(0, len(data), batch_size)
        ]
        data = torch.cat(data)

        # Randomly flip half the training data
        data[: len(data) // 2] = torch.flip(data[: len(data) // 2], [-1])

        for i in range(0, len(data), batch_size):
            # discard partial batches
            if i + batch_size > len(data):
                break

            # Slice batch from data
            inputs = data[i : i + batch_size]
            target = targets[i : i + batch_size]
            batch_count += 1

            # Compute new gradients
            train_model.zero_grad()
            train_model.train(True)

            logits = train_model(inputs)

            loss = model.label_smoothing_loss(logits, target, alpha=0.2)
            loss_value = loss.mean().item()
            loss_values = []
            loss_values.append(loss_value)

            loss.sum().backward()
            
            # Optional gradient clipping
            if config.max_grad_norm is not None:
                if config.separate_clipping:
                    # Clip each parameter separately
                    for param in train_model.parameters():
                        if param.grad is not None:
                            torch.nn.utils.clip_grad_norm_(param, config.max_grad_norm)
                else:
                    # Clip all parameters together
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), config.max_grad_norm)

            lr_index = min(batch_count, len(lr_schedule) - 1)
            lr = lr_schedule[lr_index]
            lr_bias = lr_schedule_bias[lr_index]

            # Update weights and biases of training model
            update_nesterov(weights, lr, weight_decay, momentum)
            update_nesterov(biases, lr_bias, weight_decay_bias, momentum)

            # Update validation model with exponential moving averages
            if (i // batch_size % ema_update_freq) == 0:
                update_ema(train_model, valid_model, ema_rho)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Add training time
        train_time += time.perf_counter() - start_time

        valid_correct = []
        for i in range(0, len(valid_data), batch_size):
            valid_model.train(False)

            # Test time agumentation: Test model on regular and flipped data
            regular_inputs = valid_data[i : i + batch_size]
            flipped_inputs = torch.flip(regular_inputs, [-1])

            logits1 = valid_model(regular_inputs).detach()
            logits2 = valid_model(flipped_inputs).detach()

            # Final logits are average of augmented logits
            logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)

            # Compute correct predictions
            correct = logits.max(dim=1)[1] == valid_targets[i : i + batch_size]

            valid_correct.append(correct.detach().type(torch.float64))

        # Accuracy is average number of correct predictions
        valid_acc = torch.mean(torch.cat(valid_correct)).item()

        # Store metrics
        validation_accuracies.append(valid_acc)
        avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0
        training_losses.append(avg_loss)
        
        print(f"{epoch:5} {batch_count:8d} {train_time:19.2f} {valid_acc:22.4f} {avg_loss:16.4f}")

    return validation_accuracies, training_losses

def preprocess_data(data, device, dtype):
    # Convert to torch float16 tensor
    data = torch.tensor(data, device=device).to(dtype)

    # Normalize
    mean = torch.tensor([125.31, 122.95, 113.87], device=device).to(dtype)
    std = torch.tensor([62.99, 62.09, 66.70], device=device).to(dtype)
    data = (data - mean) / std

    # Permute data from NHWC to NCHW format
    data = data.permute(0, 3, 1, 2)

    return data


def load_cifar10(device, dtype, data_dir="/run/media/robo/Data/datasets"):
    train = torchvision.datasets.CIFAR10(root=data_dir, download=True)
    valid = torchvision.datasets.CIFAR10(root=data_dir, train=False)

    train_data = preprocess_data(train.data, device, dtype)
    valid_data = preprocess_data(valid.data, device, dtype)

    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)

    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)

    return train_data, train_targets, valid_data, valid_targets


def update_ema(train_model, valid_model, rho):
    # The trained model is not used for validation directly. Instead, the
    # validation model weights are updated with exponential moving averages.
    train_weights = train_model.state_dict().values()
    valid_weights = valid_model.state_dict().values()
    for train_weight, valid_weight in zip(train_weights, valid_weights):
        if valid_weight.dtype in [torch.float16, torch.float32]:
            valid_weight *= rho
            valid_weight += (1 - rho) * train_weight


def update_nesterov(weights, lr, weight_decay, momentum):
    for weight, velocity in weights:
        if weight.requires_grad:
            gradient = weight.grad.data
            weight = weight.data

            gradient.add_(weight, alpha=weight_decay).mul_(-lr)
            velocity.mul_(momentum).add_(gradient)
            weight.add_(gradient.add_(velocity, alpha=momentum))


def random_crop(data, crop_size):
    crop_h, crop_w = crop_size
    h = data.size(2)
    w = data.size(3)
    x = torch.randint(w - crop_w, size=(1,))[0]
    y = torch.randint(h - crop_h, size=(1,))[0]
    return data[:, :, y : y + crop_h, x : x + crop_w]


def sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def getrelpath(abspath):
    import os
    return os.path.relpath(abspath, os.getcwd())


def print_info():
    # Knowing this information might improve chance of reproducability
    print("File   :", getrelpath(__file__), sha256(__file__))
    print("Model  :", getrelpath(model.__file__), sha256(model.__file__))
    print("PyTorch:", torch.__version__)


def main():
    print_info()

    accuracies = []
    threshold = 0.94
    for run in range(100):
        valid_acc = train(seed=run)
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
