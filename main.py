#!/usr/bin/env python3
"""
Main entry point for the Federated Learning Project.
Provides a command-line interface to run various experiments.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.experiments.run_centralized import main as run_centralized
from src.experiments.run_federated import main as run_federated
from src.experiments.run_model_editing import main as run_model_editing


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Federated Learning Project - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run centralized training
  python main.py centralized --model_size small --num_epochs 100
  
  # Run federated learning
  python main.py federated --num_clients 100 --num_rounds 100
  
  # Run model editing
  python main.py editing --target_sparsity 0.9 --num_iterations 10
  
  # Run with custom config
  python main.py federated --config configs/custom_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Centralized training parser
    centralized_parser = subparsers.add_parser('centralized', help='Run centralized training')
    centralized_parser.add_argument('--model_size', default='small', 
                                  choices=['tiny', 'small', 'base', 'large'])
    centralized_parser.add_argument('--num_epochs', type=int, default=100)
    centralized_parser.add_argument('--learning_rate', type=float, default=0.01)
    centralized_parser.add_argument('--batch_size', type=int, default=128)
    centralized_parser.add_argument('--device', type=str, default='auto',
                                  choices=['auto', 'mps', 'cuda', 'cpu'],
                                  help='Device to use (auto=mps>cuda>cpu)')
    centralized_parser.add_argument('--config', default='configs/default_config.yaml')
    
    # Federated learning parser
    federated_parser = subparsers.add_parser('federated', help='Run federated learning')
    federated_parser.add_argument('--num_clients', type=int, default=100)
    federated_parser.add_argument('--num_rounds', type=int, default=100)
    federated_parser.add_argument('--client_fraction', type=float, default=0.1)
    federated_parser.add_argument('--non_iid_degree', type=float, default=0.0)
    federated_parser.add_argument('--device', type=str, default='auto',
                                choices=['auto', 'mps', 'cuda', 'cpu'],
                                help='Device to use (auto=mps>cuda>cpu)')
    federated_parser.add_argument('--config', default='configs/default_config.yaml')
    
    # Model editing parser
    editing_parser = subparsers.add_parser('editing', help='Run model editing')
    editing_parser.add_argument('--target_sparsity', type=float, default=0.9)
    editing_parser.add_argument('--num_iterations', type=int, default=10)
    editing_parser.add_argument('--num_epochs', type=int, default=40)
    editing_parser.add_argument('--device', type=str, default='auto',
                              choices=['auto', 'mps', 'cuda', 'cpu'],
                              help='Device to use (auto=mps>cuda>cpu)')
    editing_parser.add_argument('--config', default='configs/default_config.yaml')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up environment
    os.environ['PYTHONPATH'] = str(Path(__file__).parent / "src")
    
    # Run appropriate experiment
    if args.command == 'centralized':
        # Set sys.argv for the centralized script
        sys.argv = [
            'run_centralized.py',
            '--model_size', args.model_size,
            '--num_epochs', str(args.num_epochs),
            '--learning_rate', str(args.learning_rate),
            '--batch_size', str(args.batch_size),
            '--config', args.config
        ]
        run_centralized()
    
    elif args.command == 'federated':
        # Set sys.argv for the federated script
        sys.argv = [
            'run_federated.py',
            '--num_clients', str(args.num_clients),
            '--num_rounds', str(args.num_rounds),
            '--client_fraction', str(args.client_fraction),
            '--non_iid_degree', str(args.non_iid_degree),
            '--config', args.config
        ]
        run_federated()
    
    elif args.command == 'editing':
        # Set sys.argv for the model editing script
        sys.argv = [
            'run_model_editing.py',
            '--target_sparsity', str(args.target_sparsity),
            '--num_iterations', str(args.num_iterations),
            '--num_epochs', str(args.num_epochs),
            '--config', args.config
        ]
        run_model_editing()


if __name__ == '__main__':
    main()
