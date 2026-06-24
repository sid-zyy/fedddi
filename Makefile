# Makefile — FedDDI centralized baseline
# ─────────────────────────────────────────────────────────────────────────────
# Common workflows:
#   make install        — install all Python dependencies
#   make prepare-data   — generate ATC mapping & precompute chemical features
#   make train          — run full training (10 runs, matches leaderboard)
#   make train-quick    — 1 run, 100 epochs (for rapid smoke-testing)
#   make train-fl       — run full FL simulation (50 rounds, 5 clients)
#   make train-fl-quick — FL dry-run smoke test (2 rounds, CPU)
#   make test           — run all unit tests
#   make test-fast      — skip slow tests
#   make export         — export embeddings from best checkpoint
#   make predict-test   — score the OGB test split with a saved checkpoint
#   make format         — black + ruff --fix
#   make lint           — ruff check + pyright
#   make clean          — remove generated artefacts
#
# NOTE: `ablation`, `sweep`, `profile`, and `visualize` are aspirational
# targets — the scripts they call (ablation.py, conf/sweep.yaml,
# scripts/profile_anchors.py, scripts/visualize_distances.py) are not part
# of this repository snapshot. The targets below print a clear message
# instead of failing with "file not found" if you run them.

.PHONY: install prepare-data train train-fl train-fl-quick train-quick ablation ablation-single sweep test test-fast \
        test-cov profile visualize export predict-test format lint clean

# ── Installation ──────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	pip install -U "flwr[simulation]"

# ── Data Preparation ──────────────────────────────────────────────────────────
prepare-data:
	python -c "from src.data.dataset import load_dataset; load_dataset(root='./data')"
	python scripts/prepare_atc_mapping.py
	python scripts/prepare_features.py

# ── Training ────────────────────────────────────────────────────────────────────
train:
	python train.py \
	  training.num_runs=10 \
	  dataset.num_anchors=1000 \
	  model.hidden_dim=500 \
	  training.epochs=500 \
	  training.lr=0.003

train-quick:
	python train.py \
	  training.num_runs=1 \
	  training.epochs=100 \
	  training.patience=20 \
	  dataset.num_anchors=200 \
	  wandb.mode=disabled

# ── Ablation (not included in this repo snapshot) ────────────────────────────
ablation:
	@echo "ablation.py is not part of this repository snapshot — skipping."
	@exit 1

ablation-single:
	@echo "ablation.py is not part of this repository snapshot — skipping."
	@exit 1

# ── Sweep (not included in this repo snapshot) ───────────────────────────────
sweep:
	@echo "conf/sweep.yaml is not part of this repository snapshot — skipping."
	@echo "You can still run a manual Hydra multirun, e.g.:"
	@echo "  python train.py --multirun training.lr=0.001,0.003,0.01"
	@exit 1

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow and not integration"

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Analysis scripts (not included in this repo snapshot) ───────────────────
profile:
	@echo "scripts/profile_anchors.py is not part of this repository snapshot — skipping."
	@exit 1

visualize:
	@echo "scripts/visualize_distances.py is not part of this repository snapshot — skipping."
	@exit 1

export:
	python scripts/export_embeddings.py \
	  --checkpoint ./checkpoints/best_run0.pt \
	  --out ./embeddings.npy

# ── Prediction ────────────────────────────────────────────────────────────────
predict-test:
	python predict.py checkpoint=./checkpoints/best_run0.pt mode=test_split

# ── Federated Learning ────────────────────────────────────────────────────────
train-fl:
	python main_fl.py \
	  num_rounds=50 \
	  local_epochs=5 \
	  num_clients=5 \
	  device=cuda

train-fl-quick:
	python main_fl.py \
	  num_rounds=2 \
	  local_epochs=1 \
	  num_clients=5 \
	  training.epochs=1 \
	  device=cpu \
	  wandb.mode=disabled

# ── Code Quality ──────────────────────────────────────────────────────────────
format:
	black .
	ruff check --fix .

lint:
	ruff check .
	pyright

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf checkpoints/ figures/ embeddings.npy outputs/ multirun/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
