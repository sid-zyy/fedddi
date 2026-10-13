#!/bin/bash
# GPU checks for a Kaggle notebook (GPU T4, Internet on). Run from the repository root:
#   bash scripts/kaggle_gpu_checks.sh [setup|smoke|epochs|lrdecay]
# setup:  install dependencies and download ogbl-ddi plus the ATC mapping
# smoke:  GPU test suite and a 2-round CAGA run (~15 min)
# epochs: clean FedAvg, 100 rounds, 2 vs. 5 local epochs on tuning seed 10 (a few hours each)
# lrdecay: the 5-epoch run again with training.round_lr_decay=true (about 2.5 hours), to compare
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
  *)
    echo "usage: bash scripts/kaggle_gpu_checks.sh [setup|smoke|epochs|lrdecay]"; exit 1
    ;;
esac
