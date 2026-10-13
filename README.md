# FedDDI-CAGA

Clean-Anchor Gated Aggregation (CAGA) for federated drug–drug interaction prediction on `ogbl-ddi`. One probe-gated aggregation rule is used against label poisoning and feature drift together, without knowing which is present.

The model combines a GraphSAGE encoder with a Hadamard-product MLP link predictor. CAGA evaluates candidate aggregate updates on a clean server probe and reduces updates that increase its ranking loss. It can gate the feature projection, graph convolutions and predictor separately.

This is a research implementation. The included synthetic tests exercise aggregation behaviour; they do not establish effectiveness on the real DDI benchmark. No benchmark results or trained checkpoints are bundled.

## Research setup

Every client uses the same public graph and fixed structural node features. Clients differ in the training interactions they sample. Poisoning changes local labels and can scale parameter updates; feature drift changes a client's feature values. CAGA reserves server training edges outside the client observation pool.

This is a shared-public-graph simulation, not a deployment on private hospital graphs. It does not implement differential privacy or secure aggregation.

FedAvg, FedProx, median-norm clipping and FLTrust are retained as comparison baselines. CAGA is selected with `defense.mode=caga`; `defense.mode=none` is the default for an undefended baseline.

## Installation

Use Python 3.10 or newer in a virtual environment. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install a PyTorch build compatible with your GPU before installing the requirements if needed. The requirements specify version ranges, not a verified lockfile.

## Prepare the data

```bash
python -c "from src.data.dataset import load_dataset; load_dataset(root='./data')"
python scripts/prepare_atc_mapping.py
```

The first command downloads `ogbl-ddi` through OGB. The second downloads therapeutic-class metadata and creates `data/ogbl_ddi/atc_mapping.pt`. Use this mapping for class-targeted poisoning experiments; without it, client observation sampling falls back to a class-agnostic split. Data is downloaded separately and excluded from Git.

## Run CAGA

Start with a short run to check the installation. It still loads and processes the full graph.

```bash
python main_fl.py defense.mode=caga num_clients=3 num_rounds=2 local_epochs=1 device=cuda wandb.mode=disabled plotting.enabled=false
```

Use `device=cpu` if a GPU is unavailable, with substantially longer runtime. To run the same setup without CAGA, change `defense.mode=caga` to `defense.mode=none`. Add `fl.algorithm=fedprox` for the FedProx optimizer or use `defense.mode=norm_clip` for the clipping baseline. `defense.mode=fltrust` runs FLTrust (Cao et al., NDSS 2021): the server trains on the same reserved root edges as CAGA each round and weights clients by ReLU-clipped cosine similarity to that update. `defense.mode=root_client` is a control: plain FedAvg with the server's root-edge update averaged in as one more client, which separates the benefit of the server's clean data from the benefit of a defense.

For a centralized baseline with the same fixed features:

```bash
python train.py baseline.use_fixed_features=true training.num_runs=1 training.epochs=50 device=cuda wandb.mode=disabled
```

Training and attack settings are in [conf/config.yaml](conf/config.yaml). Per-round metrics are written to `results_*.csv`; use `+csv_output_dir=results/pilot` to choose a destination.

## Experiments

The Python runner replaces the old notebook workflows. Preview the first seed before starting GPU jobs:

```bash
python scripts/run_experiments.py --tiers 1 --seeds 0
python scripts/run_experiments.py --tiers 1 --seeds 0 --execute
python scripts/run_experiments.py --tiers 1 2 3 --execute
python scripts/run_experiments.py --summarize
```

The full protocol uses 10 clients, 100 rounds and five seeds. Completed runs are skipped; interrupted runs restart from round one. There is no wall-clock budget limit. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for conditions, metrics and interpretation.

## Tests

```bash
python -m pytest tests/ -m "not slow and not integration"
```

To run only the lightweight CAGA tests in a NumPy environment:

```bash
python -m pip install numpy pandas pyyaml pytest
python -m pytest tests/test_caga.py tests/test_caga_sim.py tests/test_actor_cache.py tests/test_journal_protocol.py tests/test_caga_strategy.py tests/test_fltrust.py
```

Some legacy federated tests install module-level test doubles. If they interfere with an installed Flower/PyG environment, run those test files in separate processes. The full GPU pipeline must be validated with the intended dependency versions and dataset.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main_fl.py` | Flower simulation and CAGA integration |
| `train.py` | Centralized training |
| `src/training/caga.py` | NumPy aggregation algorithm |
| `src/training/caga_probe.py` | Clean-probe scoring and parameter block mapping |
| `src/training/fltrust.py` | FLTrust baseline aggregation |
| `src/data/` | Dataset, features, observation sampling and attack simulation |
| `src/models/` | GraphSAGE encoder and link predictor |
| `scripts/` | Experiment runner, data preparation and diagnostics |
| `tests/` | Algorithm and pipeline regression tests |
| `docs/` | Method and experiment documentation |
| `PUBLICATION_PLAN.md` | Thesis, status and roadmap toward a journal paper |

`predict.py` and `scripts/export_embeddings.py` support checkpoints from the learned-embedding centralized mode. Fixed-feature checkpoint loading is not implemented in those utilities.

## License

GNU GPL version 3; see [LICENSE](LICENSE). Dataset access and third-party metadata remain subject to their own terms.
