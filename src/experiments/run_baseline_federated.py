import argparse
import torch

from src.data.dataset_loader import CIFAR100DataManager
from src.training.baseline_federated_training import BaselineFederatedTrainer
from src.models.vision_transformer import DINOBackboneClassifier


def get_device(pref: str = 'auto'):
    if pref == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if pref == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if pref == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--num_clients', type=int, default=100)
    p.add_argument('--client_fraction', type=float, default=0.1)
    p.add_argument('--num_rounds', type=int, default=50)
    p.add_argument('--num_client_steps', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--freeze_backbone', action='store_true', default=True)
    p.add_argument('--classes_per_client', type=int, default=None, 
                   help='Number of classes per client (overrides non_iid_degree)')
    # Checkpointing & resume support
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                   help='Directory to save federated checkpoints')
    p.add_argument('--checkpoint_interval', type=int, default=50,
                   help='Save a checkpoint every N federated rounds')
    p.add_argument('--resume_from', type=str, default=None,
                   help='Path to a checkpoint .pth file to resume from')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    data_manager = CIFAR100DataManager(batch_size=args.batch_size, download=True)
    client_datasets = data_manager.create_federated_datasets(
        num_clients=args.num_clients, 
        classes_per_client=args.classes_per_client,
        val_split=args.val_split
    )
    _, val_loader, _ = data_manager.get_centralized_loaders(val_split=args.val_split)

    model = DINOBackboneClassifier(num_classes=100, freeze_backbone=args.freeze_backbone).to(device)

    optimizer_config = {
        'lr': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    }

    trainer = BaselineFederatedTrainer(
        model=model,
        client_datasets=client_datasets,
        optimizer_config=optimizer_config,
        device=str(device),
        num_clients=args.num_clients,
        client_fraction=args.client_fraction,
        num_client_steps=args.num_client_steps,
        batch_size=args.batch_size
    )

    history = trainer.train_federated_rounds(
        num_rounds=args.num_rounds,
        validation_loader=val_loader,
        model_name="baseline_fedavg",
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from
    )

    if history['validation_accuracies']:
        print(f"Best validation accuracy: {max(history['validation_accuracies']):.2f}%")
    else:
        print("No validation metrics recorded.")
    
    # Save results to file for plotting
    import yaml
    import os
    from datetime import datetime
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "iid" if args.classes_per_client == 100 else f"non_iid_{args.classes_per_client}"
    filename = f"federated_{scenario_name}_results_{timestamp}.yaml"
    filepath = os.path.join(results_dir, filename)
    
    # Prepare results for saving
    results = {
        'experiment_config': {
            'num_clients': args.num_clients,
            'client_fraction': args.client_fraction,
            'num_rounds': args.num_rounds,
            'num_client_steps': args.num_client_steps,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'batch_size': args.batch_size,
            'val_split': args.val_split,
            'classes_per_client': args.classes_per_client,
            'freeze_backbone': args.freeze_backbone,
            'scenario': scenario_name
        },
        'training_history': history,
        'best_validation_accuracy': max(history['validation_accuracies']) if history['validation_accuracies'] else 0.0,
        'final_validation_accuracy': history['validation_accuracies'][-1] if history['validation_accuracies'] else 0.0,
        'best_client_accuracy': max(history['round_accuracies']) if history['round_accuracies'] else 0.0,
        'final_client_accuracy': history['round_accuracies'][-1] if history['round_accuracies'] else 0.0
    }
    
    # Save to YAML file
    with open(filepath, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\nResults saved to: {filepath}")
    print(f"Best validation accuracy: {results['best_validation_accuracy']:.2f}%")
    print(f"Final validation accuracy: {results['final_validation_accuracy']:.2f}%")
    print(f"Best client accuracy: {results['best_client_accuracy']:.2f}%")
    print(f"Final client accuracy: {results['final_client_accuracy']:.2f}%")


if __name__ == "__main__":
    main()


