# FedDDI: HARBOR

HARBOR is a server-side aggregation method for federated drug–drug interaction (DDI) prediction that targets clinical safety: it aims to miss as few high-risk interactions as possible without raising false alarms. Each round, the server aggregates client updates with a robust rule (FLTrust), trains from that aggregate on its small trusted dataset enriched in high-risk pairs, and applies the resulting correction capped at the size of the clients' own update, so it steers the federation without replacing it.

The model is a GraphSAGE encoder (GCN optional) with a Hadamard-product MLP link predictor. The repository also contains the attack simulations, safety metrics and comparison defenses used to evaluate it.

This is a research implementation accompanying a manuscript that is under review and has not been accepted or published. Results and design choices may change, and the code should not be cited as a published method. No data, results or trained checkpoints are bundled.

## What is included

- **HARBOR** (`--method anchor` in the experiment scripts; the reported setting is the `anchor_h50` preset).
- **Baselines:** FedAvg, FedProx, median-norm clipping, FLTrust, LFR, ERR, Zeno, Krum, coordinate median, trimmed mean, centered clipping, Divide-and-Conquer, FLAME, a FreqFed-style filter, FedVal and CAGA (clean-anchor gated aggregation).
- **Client feature drift:** two honest clients can be given a sudden or gradual covariate shift of their node features (`src/data/data_drift.py`). The drift is injected to make clients differ; nothing in the code detects or measures it.
- **Attacks:** label poisoning, removal or fabrication of high-risk interactions, A Little Is Enough, sign-flip, Gaussian, IPM, min-max and min-sum, on-off and adaptive attackers.
- **Safety metrics** (`src/training/safety_metrics.py`): missed dangerous interaction rate (MDIR), recall by risk class and risk-weighted recall, counterfactual clinical harm (CCH) and damage recovery ratio (DRR), all at a fixed 95% specificity.

## Installation

Use Python 3.10 or newer in a virtual environment. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install a PyTorch build compatible with your GPU before installing the requirements if needed. The requirements specify version ranges, not a verified lockfile.

## Prepare the data

Datasets: [DrugBank](https://go.drugbank.com/) (licensed), [DDInter](https://ddinter.scbdd.com/) (CC BY-NC-SA 4.0) and [ogbl-ddi](https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi) (public).

**ogbl-ddi** (public, downloaded through OGB):

```bash
python -c "from src.data.dataset import load_dataset; load_dataset(root='./data')"
python scripts/prepare_atc_mapping.py
```

**DrugBank** (needed for the node features and the DrugBank-based experiments). DrugBank is licensed: obtain the full XML export from DrugBank under your own licence. It must not be redistributed, and everything built from it stays under `data/`, which is excluded from Git.

```bash
python scripts/prepare_drugbank.py --xml "path/to/full database.xml" --out data/drugbank --approved-only
```

Severity classes (low, moderate, high) are assigned from the interaction descriptions using [conf/ddi_risk_map.yaml](conf/ddi_risk_map.yaml). These outcome-type classes agree poorly with expert grades (weighted kappa 0.04 on the pairs shared with DDInter), so they are a secondary analysis.

**DDInter** (expert severity grades; used for the main safety results). Download the DDInter files from the site above, then match them to the prepared DrugBank drugs:

```bash
python scripts/prepare_ddinter.py --ddinter data/ddinter --drugbank data/drugbank --out data/ddinter_graph
```

## Run HARBOR

A single run on DrugBank (10 clients, 40 rounds). Use `--device cpu` without a GPU, at a much longer runtime.

```bash
python scripts/cpu_pilot.py --dataset drugbank --cond clean --method anchor --root-obs 20000 \
    --anchor-base fltrust --anchor-high-frac 0.5 --anchor-kappa 1.0 --lr 0.003 --local-epochs 2 \
    --seed 0 --device cuda --out results/harbor
```

Replace `--method anchor ...` with another method (for example `--method fltrust --root-obs 20000` or `--method fedavg`) to run a baseline, and change `--cond` or add attack flags to simulate an attack. `python scripts/cpu_pilot.py --help` lists every option. Per-round metrics are written to `rounds.csv` in the run's folder.

To run a grid of named method and attack presets over several seeds in parallel:

```bash
python scripts/gpu_variant_study.py --dataset drugbank --seeds 0 1 --methods fedavg fltrust anchor_h50 \
    --attacks clean hideHigh4 --device cuda --parallel 3 --out results/study
python scripts/summarize_variants.py results/study
```

The presets are defined at the top of [scripts/gpu_variant_study.py](scripts/gpu_variant_study.py). `scripts/make_kaggle_bundle.py` packages the code and builds ready-to-run Kaggle notebooks for the same grids.

## Flower simulation and centralized training

The Flower-based pipeline runs the original defenses on ogbl-ddi at full scale:

```bash
python main_fl.py defense.mode=fltrust num_clients=3 num_rounds=2 local_epochs=1 device=cuda wandb.mode=disabled plotting.enabled=false
python train.py baseline.use_fixed_features=true training.num_runs=1 training.epochs=50 device=cuda wandb.mode=disabled
```

`defense.mode` accepts `none`, `norm_clip`, `fltrust`, `root_client` and `caga`. Settings are in [conf/config.yaml](conf/config.yaml).

## Tests

```bash
python -m pytest tests/ -m "not slow and not integration"
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `scripts/cpu_pilot.py` | Experiment runner: every method (including HARBOR), attack and metric |
| `scripts/gpu_variant_study.py` | Named presets and parallel grids over seeds |
| `scripts/summarize_variants.py` | Summary tables from run folders |
| `scripts/prepare_drugbank.py`, `scripts/prepare_atc_mapping.py` | Data preparation |
| `src/training/robust_aggregation.py` | HARBOR's capped server correction and the robust-aggregation baselines |
| `src/training/safety_metrics.py` | MDIR, recall by risk class, CCH and DRR |
| `src/training/model_attacks.py` | Model-poisoning and adaptive attacks |
| `src/data/` | Datasets, features, client sampling and attack simulation |
| `src/models/` | GraphSAGE/GCN encoder and link predictor |
| `main_fl.py`, `train.py` | Flower simulation and centralized training |
| `tests/` | Regression tests |

## License

GNU GPL version 3; see [LICENSE](LICENSE). Dataset access and third-party metadata remain subject to their own terms.
