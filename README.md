## Federated Learning with ViT and TaLoS (Local)

Minimal, local-first implementation of federated learning (FedAvg) on CIFAR‑100 with Vision Transformers, plus TaLoS-based model editing and mask generation. Works on Apple Silicon via PyTorch MPS.

### Install
```bash
pip install -r requirements.txt
```

### Quick start
```bash
# Centralized training
python main.py centralized --model_size small --num_epochs 100

# Federated learning
python main.py federated --num_clients 100 --num_rounds 100

# Model editing / pruning (TaLoS)
python main.py editing --target_sparsity 0.9 --num_iterations 10
```

### Configs
Edit `configs/default_config.yaml` or choose presets in `configs/experiment_configs.yaml`.

### Project layout
- `main.py`: CLI entrypoint (centralized | federated | editing)
- `src/experiments`: runnable experiment scripts
- `src/training`: training loops, sparse optimizer, editing
- `src/models/vision_transformer.py`: ViT variants
- `src/data/dataset_loader.py`: CIFAR‑100 loading & splits
- `masks/`, `checkpoints/`, `results/`, `wandb/`: outputs (git-ignored)

### Apple Silicon (MPS)
MPS is auto-selected when available; override with `--device mps|cuda|cpu`.

### Notes
- Artifacts and files >100MB are excluded from version control.
- Dataset downloads happen automatically on first run.
