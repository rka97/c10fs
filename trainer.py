import json
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import model


def _to_numpy(x):
    return [y.item() if torch.is_tensor(y) else y for y in x]

class CifarTransform:
    def __init__(self, crop_size=(32, 32)):
        self.crop_size = crop_size

    def __call__(self, batch):
        data, targets = batch
        # Random crop
        data = self._random_crop(data, self.crop_size)
        # Random horizontal flip of first half
        data[: len(data) // 2] = torch.flip(data[: len(data) // 2], [-1])
        return data, targets

    @staticmethod
    def _random_crop(data, crop_size=(32, 32)):
        crop_h, crop_w = crop_size
        h = data.size(2)
        w = data.size(3)
        x = torch.randint(w - crop_w, size=(1,))[0]
        y = torch.randint(h - crop_h, size=(1,))[0]
        return data[:, :, y : y + crop_h, x : x + crop_w]


class Trainer:
    def __init__(self, config):
        self.config = config
        self.transform = CifarTransform()

    def _get_lr_schedule(self, optimizer_name):
        if optimizer_name == "sgd":
            return torch.cat(
                [
                    torch.linspace(0, self.config.lr_max, self.config.warmup_steps),
                    torch.linspace(
                        self.config.lr_max,
                        self.config.lr_final,
                        self.config.decay_steps,
                    ),
                ]
            )
        elif optimizer_name == "adamw":
            return torch.cat(
                [
                    torch.linspace(
                        0, self.config.lr_max * 10, self.config.warmup_steps
                    ),
                    torch.linspace(
                        self.config.lr_max * 10,
                        self.config.lr_final,
                        self.config.decay_steps,
                    ),
                ]
            )

    def preprocess_data(self, data):
        data = torch.tensor(data, device=self.config.device).to(self.config.dtype)
        mean = torch.tensor([125.31, 122.95, 113.87], device=self.config.device).to(
            self.config.dtype
        )
        std = torch.tensor([62.99, 62.09, 66.70], device=self.config.device).to(
            self.config.dtype
        )
        data = (data - mean) / std
        data = data.permute(0, 3, 1, 2)  # Permute data from NHWC to NCHW format
        return data

    def prepare_data(self):
        train = torchvision.datasets.CIFAR10(root=self.config.data_dir, download=True)
        valid = torchvision.datasets.CIFAR10(root=self.config.data_dir, train=False)

        train_data = self.preprocess_data(train.data)
        valid_data = self.preprocess_data(valid.data)

        train_targets = torch.tensor(train.targets, device=self.config.device)
        valid_targets = torch.tensor(valid.targets, device=self.config.device)

        # Pad training data
        train_data = nn.ReflectionPad2d(4)(train_data)

        train_dataset = TensorDataset(train_data, train_targets)
        valid_dataset = TensorDataset(valid_data, valid_targets)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        return train_dataset

    def update_ema(self, train_model, valid_model, rho):
        train_weights = train_model.state_dict().values()
        valid_weights = valid_model.state_dict().values()
        for train_weight, valid_weight in zip(train_weights, valid_weights):
            if valid_weight.dtype in [torch.float16, torch.float32]:
                valid_weight *= rho
                valid_weight += (1 - rho) * train_weight

    def _update_lr(self, optimizer, batch_count):
        lr_index = min(batch_count, len(self.lr_schedule) - 1)
        current_lr = self.lr_schedule[lr_index]

        if isinstance(optimizer, torch.optim.SGD):
            # For SGD, we keep the separate learning rates for weights and biases
            for i, param_group in enumerate(optimizer.param_groups):
                # First group is weights, second is biases
                param_group["lr"] = current_lr * (64.0 if i == 1 else 1.0)
        elif isinstance(optimizer, torch.optim.AdamW):
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = current_lr * (64.0 if i == 1 else 1.0)
                # param_group['lr'] = 1e-6
        else:
            # For others, use a single learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

    def train(self, optimizer_name, seed=0):
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

        start_time = time.perf_counter()

        # Initialize dataset and model
        train_dataset = self.prepare_data()
        weights = model.patch_whitening(train_dataset.tensors[0][:10000, :, 4:-4, 4:-4])
        train_model = model.Model(weights, c_in=3, c_out=10, scale_out=0.125)
        train_model.to(self.config.dtype)

        # Handle BatchNorm precision
        for module in train_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.float()

        train_model.to(self.config.device)
        valid_model = copy.deepcopy(train_model)

        # Initialize optimizer
        if optimizer_name == "sgd":
            optimizer = self._init_sgd(train_model)
        elif optimizer_name == "adamw":
            optimizer = self._init_adamw(train_model)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.lr_schedule = self._get_lr_schedule(optimizer_name)

        print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")
        print(
            "\nepoch    batch    train time [sec]    train loss    validation accuracy    learning rate"
        )

        train_time = 0.0
        batch_count = 0
        train_losses = []
        valid_accs = []
        learning_rates = []

        for epoch in range(1, self.config.epochs + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Training
            train_model.train()
            epoch_losses = []
            for data, target in self.train_loader:
                data, target = self.transform((data, target))
                batch_count += 1

                # Update learning rate
                self._update_lr(optimizer, batch_count)
                current_lr = optimizer.param_groups[0]["lr"]
                learning_rates.append(current_lr)

                optimizer.zero_grad()
                logits = train_model(data)
                loss = model.label_smoothing_loss(logits, target, alpha=0.2)

                # # Add gradient clipping for AdamW
                # if optimizer_name == "adamw":
                #     torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)

                loss_val = loss.sum().item()
                epoch_losses.append(loss_val)
                loss.sum().backward()
                optimizer.step()

                if batch_count % self.config.ema_update_freq == 0:
                    self.update_ema(train_model, valid_model, self.config.ema_rho)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_time += time.perf_counter() - start_time
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_train_loss)

            # Validation
            valid_correct = []
            valid_model.eval()
            with torch.no_grad():
                for data, target in self.valid_loader:
                    data, target = data.to(self.config.device), target.to(
                        self.config.device
                    )
                    logits1 = valid_model(data)
                    logits2 = valid_model(torch.flip(data, [-1]))
                    logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)
                    correct = logits.max(dim=1)[1] == target
                    valid_correct.append(correct.type(torch.float64))

            valid_acc = torch.mean(torch.cat(valid_correct)).item()
            valid_accs.append(valid_acc)

            print(
                f"{epoch:5} {batch_count:8d} {train_time:19.2f} {avg_train_loss:13.4f} "
                f"{valid_acc:22.4f} {current_lr:16.6f}"
            )
        results = {
            "train_losses": _to_numpy(train_losses),
            "valid_accs": _to_numpy(valid_accs),
            "learning_rates": _to_numpy(learning_rates),
            "final_acc": valid_acc,
        }
        with open(f"training_results_{optimizer_name}.json", "w") as f:
            json.dump(results, f)

        return results

    def _init_sgd(self, model):
        weights_params = [
            p for p in model.parameters() if p.requires_grad and len(p.shape) > 1
        ]
        bias_params = [
            p for p in model.parameters() if p.requires_grad and len(p.shape) <= 1
        ]

        optimizer = torch.optim.SGD(
            [
                {
                    "params": weights_params,
                    "lr": 0.0,
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": bias_params,
                    "lr": 0.0,
                    "weight_decay": self.config.weight_decay_bias,
                },
            ],
            momentum=self.config.momentum,
            nesterov=True,
        )

        return optimizer

    def _init_adamw(self, model):
        weights_params = [
            p for p in model.parameters() if p.requires_grad and len(p.shape) > 1
        ]
        bias_params = [
            p for p in model.parameters() if p.requires_grad and len(p.shape) <= 1
        ]
        return torch.optim.AdamW(
            [
                {
                    "params": weights_params,
                    "lr": 0.0,
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": bias_params,
                    "lr": 0.0,
                    "weight_decay": self.config.weight_decay_bias,
                },
            ],
            betas=(0.9, 0.999),
        )
