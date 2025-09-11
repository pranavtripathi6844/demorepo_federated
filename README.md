Federated ViT + TaLoS: Brief Project Flow
Overview: Local-first federated learning on CIFAR‑100 with Vision Transformers. Supports centralized training, FedAvg, and TaLoS-based model editing (masking/pruning). Outputs (checkpoints, masks, metrics, plots) are generated locally and ignored in git.
Core ideas:
Vision Transformer backbone in src/models/vision_transformer.py
FedAvg with configurable client sampling and IID/non‑IID splits
TaLoS model editing: Fisher-informed importance scores → iterative pruning masks → sparse training
Runs on Apple Silicon via PyTorch MPS (or CPU/CUDA)
Project flow (end-to-end):
1) Data + device setup
CIFAR‑100 auto-download on first run
Device auto-selects MPS → CUDA → CPU (override with --device)
2) Stage 1 — Mask computation (editing)
Compute parameter importance (Fisher/criteria), generate masks at target sparsity
Iterative refinement across rounds for stability (soft-zeroing)
Entry: python main.py editing --target_sparsity 0.9 --num_iterations 10
Stores masks under masks/ (git-ignored)
3) Stage 2 — Training with/without masks
Centralized baseline: python main.py centralized --model_size small --num_epochs 100
Federated FedAvg: python main.py federated --num_clients 100 --num_rounds 100
If masks are present, training can enforce sparsity during forward/optimizer steps
Checkpoints/metrics logged to checkpoints/, results/, and optionally wandb/
4) Evaluation + analysis
Plot and compare curves/masks via scripts in src/experiments and top-level plotting utilities
Metrics saved as YAML/CSV and plots as PNGs (all git-ignored)
How to run (common commands):
Centralized: python main.py centralized --model_size small --num_epochs 100
Federated: python main.py federated --num_clients 100 --num_rounds 100
Editing (TaLoS): python main.py editing --target_sparsity 0.9 --num_iterations 10
Use --config to point to configs/default_config.yaml or presets in configs/experiment_configs.yaml
Non‑IID support:
Configure client data heterogeneity (e.g., label-skew) in federated runs via flags/configs
Standard and custom non‑IID setups available in src/experiments
Key structure:
main.py: CLI entry (centralized | federated | editing)
src/experiments: runnable experiment drivers
src/training: loops, FedAvg, sparse optimizer, editing logic
src/models/vision_transformer.py: ViT variants (tiny/small/base/large)
src/data/dataset_loader.py: CIFAR‑100 loading and splits
configs/: default and experiment configs
Outputs: masks/, checkpoints/, results/, plots/, wandb/ (ignored in git)
Apple Silicon:
MPS is auto-used if available. Override with --device mps|cuda|cpu.
