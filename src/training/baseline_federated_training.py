import copy
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os


class BaselineFederatedTrainer:
    def __init__(self,
                 model: nn.Module,
                 client_datasets: List[torch.utils.data.Dataset],
                 optimizer_config: Dict[str, Any],
                 device: str = 'cuda',
                 num_clients: int = 100,
                 client_fraction: float = 0.1,
                 num_client_steps: int = 4,
                 batch_size: int = 64,
                 checkpoint_dir: str = "./checkpoints",
                 checkpoint_interval: int = 50):
        self.global_model = model
        self.client_datasets = client_datasets
        self.optimizer_config = optimizer_config
        self.device = device
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.num_client_steps = num_client_steps
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        torch.manual_seed(42)
        np.random.seed(42)

    def train_federated_rounds(self,
                               num_rounds: int,
                               validation_loader: DataLoader,
                               model_name: str = "baseline_fedavg",
                               checkpoint_dir: str = None,
                               checkpoint_interval: int = None,
                               resume_from: str = None) -> Dict[str, List[float]]:
        history = {
            'round_losses': [],
            'round_accuracies': [],
            'validation_losses': [],
            'validation_accuracies': [],
            'selected_clients': []
        }
        # Resolve checkpoint settings
        ckpt_dir = checkpoint_dir or self.checkpoint_dir
        ckpt_int = checkpoint_interval or self.checkpoint_interval
        os.makedirs(ckpt_dir, exist_ok=True)

        # Resume if requested
        start_round = 0
        if resume_from is not None and os.path.exists(resume_from):
            try:
                payload = torch.load(resume_from, map_location='cpu')
                if isinstance(payload, dict):
                    if 'model_state' in payload:
                        self.global_model.load_state_dict(payload['model_state'])
                    if 'history' in payload and isinstance(payload['history'], dict):
                        history = payload['history']
                    start_round = int(payload.get('round', 0))
                    print(f"Resuming from checkpoint: {resume_from} at round {start_round}")
            except Exception as e:
                print(f"Failed to resume from {resume_from}: {e}")

        for round_idx in range(start_round, num_rounds):
            selected = self._select_clients()
            history['selected_clients'].append(selected)

            client_states, client_metrics, client_sizes = self._train_selected_clients(selected)

            self._aggregate_fedavg(client_states, client_sizes)

            val_loss, val_acc = self._evaluate(self.global_model, validation_loader)
            avg_client_loss = float(np.mean([m['loss'] for m in client_metrics])) if client_metrics else 0.0
            avg_client_acc = float(np.mean([m['accuracy'] for m in client_metrics])) if client_metrics else 0.0

            history['round_losses'].append(avg_client_loss)
            history['round_accuracies'].append(avg_client_acc)
            history['validation_losses'].append(val_loss)
            history['validation_accuracies'].append(val_acc)

            print(f"[Round {round_idx+1}/{num_rounds}] "
                  f"ClientLoss={avg_client_loss:.4f} ClientAcc={avg_client_acc:.2f}% "
                  f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}%")

            # Periodic checkpoint
            if ckpt_int and (round_idx + 1) % ckpt_int == 0:
                self._save_checkpoint(
                    round_completed=round_idx + 1,
                    history=history,
                    model_name=model_name,
                    checkpoint_dir=ckpt_dir
                )

        return history

    def _save_checkpoint(self,
                         round_completed: int,
                         history: Dict[str, List[float]],
                         model_name: str,
                         checkpoint_dir: str) -> None:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            filename = f"{model_name}_round_{round_completed}.pth"
            path = os.path.join(checkpoint_dir, filename)
            payload = {
                'round': round_completed,
                'model_state': self.global_model.state_dict(),
                'history': history
            }
            torch.save(payload, path)
            print(f"Saved checkpoint: {path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def _select_clients(self) -> List[int]:
        num_selected = max(int(self.num_clients * self.client_fraction), 1)
        return np.random.choice(range(self.num_clients), num_selected, replace=False).tolist()

    def _train_selected_clients(self, selected_clients: List[int]) -> Tuple[List[Dict], List[Dict], List[int]]:
        client_states, client_metrics, client_sizes = [], [], []

        for cid in selected_clients:
            loader = DataLoader(self.client_datasets[cid], batch_size=self.batch_size, shuffle=True)
            local_state, loss, acc, size = self._train_single_client(loader)
            client_states.append(local_state)
            client_metrics.append({'client_id': cid, 'loss': loss, 'accuracy': acc})
            client_sizes.append(size)

        return client_states, client_metrics, client_sizes

    def _train_single_client(self, client_loader: DataLoader) -> Tuple[Dict, float, float, int]:
        model = copy.deepcopy(self.global_model).to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.optimizer_config['lr'],
            momentum=self.optimizer_config.get('momentum', 0.9),
            weight_decay=self.optimizer_config.get('weight_decay', 0.0)
        )

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        data_iter = iter(client_loader)

        for _ in range(self.num_client_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(client_loader)
                images, labels = next(data_iter)

            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        avg_loss = total_loss / total if total > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0

        return model.state_dict(), avg_loss, acc, total

    def _aggregate_fedavg(self, client_states: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> None:
        if not client_states:
            return
        total = float(sum(client_sizes))
        keys = client_states[0].keys()
        aggregated = {}
        for k in keys:
            stacked = torch.stack([state[k].to('cpu') * (sz / total) for state, sz in zip(client_states, client_sizes)])
            aggregated[k] = stacked.sum(0)
        self.global_model.load_state_dict(aggregated)

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return (total_loss / total if total > 0 else 0.0,
                100.0 * correct / total if total > 0 else 0.0)


