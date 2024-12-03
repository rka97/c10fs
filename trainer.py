import json
import numpy as np
import time
import copy
import torch
from torch._dynamo import config
config.suppress_errors = True
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import model as model_lib


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

        self.epoch_length = len(train_data) // (self.config.batch_size * self.config.num_clients)
        print(f"Epoch length: {self.epoch_length}")

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
        return current_lr

    def _init_outer_optimizer(self, model):
        """Initialize the outer optimizer"""
        if self.config.outer_optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.config.outer_lr)
        elif self.config.outer_optimizer == "sgd_nesterov":
            return torch.optim.SGD(model.parameters(), lr=self.config.outer_lr, momentum=self.config.outer_momentum, nesterov=True)
        elif self.config.outer_optimizer == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=self.config.outer_lr)
        else:
            raise ValueError(f"Unknown outer optimizer: {self.config.outer_optimizer}")

    def compute_parameter_delta(self, prev_params, current_models):
        """Compute the parameter delta between current average and previous parameters"""
        # Get current average parameters
        current_params = []
        for param_lists in zip(*[list(model.parameters()) for model in current_models]):
            avg_param = torch.mean(torch.stack(param_lists), dim=0)
            current_params.append(avg_param)

        # Compute delta
        deltas = []
        for prev, curr in zip(prev_params, current_params):
            deltas.append(curr - prev)

        return deltas

    def apply_outer_update(self, model, delta_params):
        """Apply the outer optimization update"""
        for param, delta in zip(model.parameters(), delta_params):
            if param.requires_grad:
                param.grad = -delta  # Negative because we want to move in the delta direction

    def perform_local_steps(self, model, optimizer, num_steps, batch_count):
        """Perform local training steps for a single client

        Args:
            model: The client's model
            optimizer: The client's optimizer
            num_steps: Number of local steps to perform
            batch_count: Current batch count for learning rate scheduling

        Returns:
            list: Loss values for each step
            model: Updated model after local steps
        """
        losses = []
        learning_rates = []

        for h in range(num_steps):
            try:
                data, target = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                data, target = next(self.train_iterator)


            # Update learning rate
            lr = self._update_lr(optimizer, batch_count+h)
            learning_rates.append(lr)

            data, target = self.transform((data, target))
            optimizer.zero_grad()
            logits = model(data)
            loss = model_lib.label_smoothing_loss(logits, target, alpha=0.2)
            loss_val = loss.sum().item()
            losses.append(loss_val)
            loss.sum().backward()
            optimizer.step()

        return losses, model, learning_rates

    def train(self, seed=0, rank=0, world_size=1):
        outer_opt = None
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

        start_time = time.perf_counter()

        # Initialize dataset and models for each client
        train_dataset = self.prepare_data()
        weights = model_lib.patch_whitening(train_dataset.tensors[0][:10000, :, 4:-4, 4:-4])
        models = []
        for _ in range(self.config.num_clients):
            if self.config.test_config:
                model_instance = model_lib.SmallResNetBagOfTricks(weights, c_in=3, c_out=10, scale_out=0.125)
            else:
                model_instance = model_lib.Model(weights, c_in=3, c_out=10, scale_out=0.125)
            model_instance.to(self.config.dtype)
            for module in model_instance.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.float()
            model_instance.to(self.config.device)
            # Optionally compile the model for faster training
            if self.config.compile_model:
                model_instance = torch.compile(model_instance)
            models.append(model_instance)

        # Initialize valid_model
        valid_model = copy.deepcopy(models[0])
        valid_model.eval()

        # Initialize model with DDP
        if world_size > 1:
            models = [DDP(model, device_ids=[rank]) for model in models]

        # Initialize optimizer
        if self.config.optimizer == "sgd":
            optimizers = [self._init_sgd(model) for model in models]
        elif self.config.optimizer == "adamw":
            optimizers = [self._init_adamw(model) for model in models]
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        self.lr_schedule = self._get_lr_schedule(self.config.optimizer)
        # Initialize outer optimizer
        outer_opt = self._init_outer_optimizer(models[0])

        print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")
        # FIXME we currently count batches, there should be a better counting method
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
            for model in models:
                model.train()
            epoch_losses = []

            # Create main iterator
            self.train_iterator = iter(self.train_loader)

            for _ in range(self.epoch_length):
                # Store previous round's parameters
                prev_params = [param.detach().clone() for param in models[0].parameters()]

                # Perform local steps for each client
                all_step_losses = np.zeros((self.config.num_clients, self.config.local_steps))
                all_learning_rates = np.zeros((self.config.num_clients, self.config.local_steps))
                for k, (model, optimizer) in enumerate(zip(models, optimizers)):
                    step_losses, models[k], learning_rates = self.perform_local_steps(
                        model,
                        optimizer,
                        self.config.local_steps,
                        batch_count
                    )
                    all_step_losses[k, :] = step_losses
                    all_learning_rates[k, :] = learning_rates

                # Average losses across clients and extend to epoch losses
                epoch_losses.extend(np.mean(all_step_losses, axis=0))
                learning_rates.extend(np.mean(all_learning_rates, axis=0))

                # Update batch count and learning rate tracking
                batch_count += self.config.local_steps

                # Apply outer optimization step
                outer_opt.zero_grad()
                # Compute parameter delta and get new average
                delta_params = self.compute_parameter_delta(prev_params, models)
                self.apply_outer_update(models[0], delta_params)
                outer_opt.step()

                # Synchronize all models to the result of outer optimization
                with torch.no_grad():
                    for model in models[1:]:
                        for p_target, p_source in zip(model.parameters(), models[0].parameters()):
                            p_target.copy_(p_source)

                if batch_count % self.config.ema_update_freq == 0:
                    self.update_ema(models[0], valid_model, self.config.ema_rho)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_time += time.perf_counter() - start_time
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_train_loss)

            # Validation
            valid_correct = []
            models[0].eval()
            with torch.no_grad():
                for data, target in self.valid_loader:
                    data, target = data.to(self.config.device), target.to(
                        self.config.device
                    )
                    logits1 = models[0](data)
                    logits2 = models[0](torch.flip(data, [-1]))
                    logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)
                    correct = logits.max(dim=1)[1] == target
                    valid_correct.append(correct.type(torch.float64))

            valid_acc = torch.mean(torch.cat(valid_correct)).item()
            valid_accs.append(valid_acc)

            print(
                f"{epoch:5} {batch_count:8d} {train_time:19.2f} {avg_train_loss:13.4f} "
                f"{valid_acc:22.4f} {learning_rates[-1]:16.6f}"
            )
        results = {
            "train_losses": _to_numpy(train_losses),
            "valid_accs": _to_numpy(valid_accs),
            "learning_rates": _to_numpy(learning_rates),
            "final_acc": valid_accs[-1],
        }
        with open(f"training_results_{self.config.optimizer}.json", "w") as f:
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
