#!/usr/bin/env python3
"""
Run best-known configs for StepLR and CosineAnnealingLR for 30 epochs and
save separate plots for test loss and test accuracy for each scheduler.
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt


# Ensure project root is on path for direct script execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hyperparameter_search import HyperparameterSearch


def run_and_plot(scheduler: str,
                 learning_rate: float,
                 weight_decay: float,
                 momentum: float,
                 device: str,
                 out_dir: str,
                 val_split: float = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    hs = HyperparameterSearch(device=device, freeze_backbone=True, scheduler_type=scheduler, val_split=val_split)
    test_loss, test_acc, hist = hs.test_best_configuration({
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'scheduler_type': scheduler,
    })

    losses = hist['test_loss_history']
    accs = hist['test_accuracy_history']
    epochs = list(range(1, len(accs) + 1))

    # Save separate figures: loss and accuracy
    loss_path = os.path.join(out_dir, f"{scheduler.lower()}_test_loss.png")
    acc_path = os.path.join(out_dir, f"{scheduler.lower()}_test_accuracy.png")

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, color='tab:red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(f'{scheduler}: Test Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, accs, color='tab:blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'{scheduler}: Test Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    # Optional: also write CSVs for further analysis
    csv_path = os.path.join(out_dir, f"{scheduler.lower()}_test_metrics.csv")
    with open(csv_path, 'w') as f:
        f.write('epoch,test_loss,test_accuracy\n')
        for e, (l, a) in enumerate(zip(losses, accs), start=1):
            f.write(f'{e},{l},{a}\n')

    print(f"{scheduler}: final Test Acc={test_acc:.2f}% | Test Loss={test_loss:.4f}")
    print(f"Saved: {loss_path}, {acc_path}, {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot best StepLR and CosineAnnealingLR test curves')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Directory to save plots/CSVs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for best config')
    parser.add_argument('--wd', type=float, default=0.001, help='Weight decay for best config')
    parser.add_argument('--momentum', type=float, default=0.8, help='Momentum for best config')
    parser.add_argument('--val_split', type=float, default=None, 
                       help='Validation split ratio (e.g., 0.2 for 20% validation = 40k train, 10k val)')

    args = parser.parse_args()

    # StepLR
    run_and_plot('StepLR', args.lr, args.wd, args.momentum, args.device, args.out_dir, args.val_split)

    # CosineAnnealingLR
    run_and_plot('CosineAnnealingLR', args.lr, args.wd, args.momentum, args.device, args.out_dir, args.val_split)


if __name__ == '__main__':
    main()


