#!/bin/bash
# GPU checks for a Kaggle notebook (GPU T4, Internet on). Run from the repository root:
#   bash scripts/kaggle_gpu_checks.sh [setup|smoke|epochs|lrdecay|centralized|centralized_emb|drugbank_smoke]
# setup:  install dependencies and download ogbl-ddi plus the ATC mapping
# smoke:  GPU test suite and a 2-round CAGA run (~15 min)
# epochs: clean FedAvg, 100 rounds, 2 vs. 5 local epochs on tuning seed 10 (a few hours each)
# lrdecay: the 5-epoch run again with training.round_lr_decay=true (about 2.5 hours), to compare
# centralized: no federation, same fixed features, 3 runs (the ceiling for federated scores)
# centralized_emb: no federation, learned node embeddings, 1 run (compare with the OGB leaderboard)
# drugbank_smoke: 2-round CAGA run on DrugBank; needs data/drugbank/drugbank_ddi.pt (private, licensed data)
# Results go to /kaggle/working/results so they survive the session; download them from the Output tab.
set -euo pipefail
OUT=${OUT:-/kaggle/working/results}
mkdir -p "$OUT"
COMMON="num_clients=10 local_observations.observations_per_client=200000 local_observations.dirichlet_alpha=0.5 \
device=cuda wandb.mode=disabled plotting.enabled=false perf.dense_adj=true perf.fast_negatives=true"

case "${1:-}" in
  setup)
    pip install -q torch-geometric ogb "flwr[simulation]>=1.10" hydra-core omegaconf
    python -c "from src.data.dataset import load_dataset; load_dataset(root='./data')"
    python scripts/prepare_atc_mapping.py
    ;;
  smoke)
    # Legacy test files install module-level test doubles, so each file runs in its own process.
    for t in tests/test_*.py; do python -m pytest "$t" -q -m "not slow and not integration" -p no:cacheprovider | tail -1; done
    python main_fl.py defense.mode=caga num_clients=3 num_rounds=2 local_epochs=1 device=cuda \
      wandb.mode=disabled plotting.enabled=false +csv_output_dir="$OUT/smoke"
    ;;
  epochs)
    for EP in 2 5; do
      python main_fl.py $COMMON num_rounds=100 local_epochs=$EP training.seed=10 \
        +csv_output_dir="$OUT/clean_fedavg_ep$EP" 2>&1 | tee "$OUT/clean_fedavg_ep$EP.log"
    done
    ;;
  lrdecay)
    python main_fl.py $COMMON num_rounds=100 local_epochs=5 training.seed=10 training.round_lr_decay=true       +csv_output_dir="$OUT/clean_fedavg_ep5_lrdecay" 2>&1 | tee "$OUT/clean_fedavg_ep5_lrdecay.log"
    ;;
  centralized)
    python train.py baseline.use_fixed_features=true training.num_runs=3 training.epochs=300 training.eval_interval=5       training.patience=50 device=cuda wandb.mode=disabled plotting.enabled=false training.save_dir="$OUT/ckpt_fixed"       2>&1 | tee "$OUT/centralized_fixed.log"
    ;;
  centralized_emb)
    python train.py baseline.use_fixed_features=false training.num_runs=1 training.epochs=300 training.eval_interval=5       training.patience=50 device=cuda wandb.mode=disabled plotting.enabled=false training.save_dir="$OUT/ckpt_emb"       2>&1 | tee "$OUT/centralized_embedding.log"
    ;;
  drugbank_smoke)
    [ -f data/drugbank/drugbank_ddi.pt ] || { echo "data/drugbank/drugbank_ddi.pt missing: attach the private DrugBank dataset"; exit 1; }
    python main_fl.py dataset.name=drugbank dataset.split=random defense.mode=caga num_clients=3 num_rounds=2       local_epochs=1 device=cuda wandb.mode=disabled plotting.enabled=false perf.dense_adj=true defense.root_device=cuda       +csv_output_dir="$OUT/drugbank_smoke" 2>&1 | tee "$OUT/drugbank_smoke.log"
    ;;
  *)
    echo "usage: bash scripts/kaggle_gpu_checks.sh [setup|smoke|epochs|lrdecay|centralized|centralized_emb|drugbank_smoke]"; exit 1
    ;;
esac
