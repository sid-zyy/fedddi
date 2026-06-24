# FedDDI-GNN — Federated GNNs for Drug-Drug Interaction Prediction

A **production-ready, reproducible** implementation of the
**GraphSAGE + Hadamard MLP** baseline for the `ogbl-ddi` link prediction benchmark,
with full support for federated learning, heterogeneous per-client observations,
and concept drift simulation.

> **Recent change — Shared Global Graph (no partitioning).** Every FL client
> now trains on the SAME complete, public DDI graph (`graph = global_graph`
> for every client) — there is no client-specific node/edge partitioning in
> this project anymore. Client heterogeneity instead comes from each
> client's own private local clinical observations (which training
> interactions it samples), not from owning a different slice of the graph.
> See "Aggregation & Graph Sharing" below.

## Architecture

```
Embedding(4267, 500)           → h^0  [N, 500]
     │
SAGEConv(500, 500) + ReLU + Dropout   → h^1
     │
SAGEConv(500, 500)             → h^L  [N, 500]
     │
h_u ⊙ h_v  →  MLP(500→512→512→1)   →  logit
```

No distance encoding in this phase. The strong result (~0.80+ Val Hits@20)
comes from the Hadamard MLP decoder, 500-dim embeddings, correct BUG4 fix
(val edges included in test GNN pass), and gradient clipping.

## Key design decisions

| Choice | Value | Why |
|--------|-------|-----|
| `normalize=False` | off | L2 norm kills Hadamard signal (250× weaker) |
| `num_neg_samples=1` | 1:1 | 3:1 ratio causes gradient imbalance |
| `full_edge_index` for test | train+val | Matches DLGNN; test graph should be denser |
| `grad_clip=1.0` | 1.0 | 4267×500 embedding causes gradient explosions |
| `local_epochs` | 10 | 500 total local epochs/client (was 1, causing severe undertraining) |

## Federated Learning Performance & Heterogeneity

### Baseline Results (2 clients, alpha=0.5 non-IID)

**FedAvg 100 rounds with `num_clients=2`:**

| Setting | Val Hits@20 | Test Hits@20 | Notes |
|---------|-------------|--------------|-------|
| Centralized — fixed features | 0.56+ | 0.15–0.20 | Upper bound (full graph, no heterogeneity) |
| FedAvg — 100 rounds (stable) | 0.17–0.22 | 0.18–0.22 | Honest federated with heterogeneous observations |
| **Federation overhead** | **~0.34–0.39** | — | Gap between centralized and federated |

**Key observation:** The federated model exhibits **high-variance oscillation** (val Hits@20 swings from 0.13 to 0.20+ per round) rather than the smooth monotonic convergence of the centralized baseline. This oscillation is caused by **persistent non-IID heterogeneity**, not concept drift: with `dirichlet_alpha=0.5`, each client specializes in a different set of therapeutic areas (e.g., Client 0: cardiovascular 40%, Client 1: oncology 40%) throughout the entire run. Their locally-trained gradients point in orthogonal directions, and FedAvg averaging these divergent updates creates incoherent parameter changes. This is the baseline federation problem that FedDrift must solve even before concept drift enters the picture.

### Why More Clients Doesn't Help (with alpha=0.5)

**Short answer:** More clients = worse, not better, under current heterogeneity settings.

**Why:**

1. **Non-IID problem scales with client count.** With 5 clients at alpha=0.5, you have 5 increasingly orthogonal gradient directions. FedAvg averaging becomes even more incoherent. The oscillation grows; convergence slows further.

2. **This is not a bug — it's the research problem.** The heterogeneous-observations setting (`dirichlet_alpha=0.5`) intentionally models real hospitals with different patient populations. The federation overhead you see (0.34–0.39) is the **true cost of non-IID learning**, which **FedDrift is designed to address** by detecting divergence and adapting aggregation accordingly.

3. **More clients does NOT provide implicit regularization here.** Unlike some settings, averaging over divergent non-IID gradients doesn't regularize — it noise-injects.

### Non-IID vs. Concept Drift: Two Separate Challenges

**Baseline (stable FedAvg, no drift events):** You see ~0.34–0.39 federation overhead due to **persistent heterogeneity alone**. Each client's observation distribution is fixed for the entire 100-round run, but they are different from each other. The oscillation is steady-state non-IID, not dynamic drift.

**With concept drift (rounds 40 & 50):** On top of the baseline non-IID problem, the distributions **change mid-run**:
- **Round 40:** Client 0 suddenly shifts from one therapeutic mix to another (e.g., 16% class 8 → 40% class 0). Another perturbation on top of divergence.
- **Rounds 50–60:** Client 1 gradually transitions over 10 rounds. Continuous shift while baseline divergence persists.

**What FedDrift must solve:**

1. **Baseline:** Reduce the ~0.34 non-IID overhead in the *stable* case (no drift), outperforming vanilla FedAvg on the heterogeneous-but-static problem.
2. **Drift:** Detect distribution shifts and adapt aggregation so the model recovers faster when drift fires, minimizing the nadir and shortening recovery time.

The research contribution is **detecting and adapting to both**: FedDrift should show (a) better convergence than FedAvg even in the stable baseline, and (b) faster recovery from concept drift events than the stable baseline. If FedDrift only helps with drift but not the baseline, it's not a complete solution.

### What Would Improve Federated Performance

**Option A — Reduce heterogeneity (NOT recommended for the paper)**
```bash
python main_fl.py num_rounds=100 local_epochs=10 num_clients=5 \
    local_observations.dirichlet_alpha=2.0  # more uniform across clients
```
With `alpha=2.0`, client distributions are nearly uniform. FedAvg converges faster and reaches ~0.35–0.40 val Hits@20 on 5 clients. But this **defeats the research goal**: you're modeling hospitals as identical, not specialized.

**Option B — Use robust federated algorithms (roadmap)**
- **FedProx:** Adds a proximal term to client objectives to penalize divergence from global weights. Should reduce oscillation.
- **SCAFFOLD:** Tracks and corrects client drift via control variates.
- **FedDrift (novel):** Detects concept drift and adapts aggregation dynamically.

These algorithms keep the heterogeneous setting (honest federation) while reducing the non-IID penalty. FedDrift, specifically, should also detect when concept drift events fire and adapt the aggregation to recover faster from the resulting performance dips.

**Option C — Personalized FL (future work)**
Instead of averaging all clients' weights, learn per-client personalized models that share a foundation. Allows clients to specialize without forcing agreement on a global model.

## Repository Structure

```
├── conf/                     # Hydra configurations
│   └── config.yaml           # Default hyperparameters and run settings
├── scripts/                  # Data preparation & utility scripts
│   ├── prepare_atc_mapping.py # Maps drug IDs to ATC Level-1 codes
│   ├── prepare_features.py   # Extracts molecular fingerprints & descriptors
│   └── export_embeddings.py  # Exports GNN node representations
├── src/                      # Source code
│   ├── data/                 # Data loading
│   │   ├── dataset.py        # OGB dataset loader & PyTorch >=2.6 compatibility patch
│   │   ├── features.py       # Fixed structural feature computation (anchors + spectral)
│   │   ├── drift.py          # Concept drift simulator for client-specific distribution changes
│   │   └── local_observations.py # Heterogeneous per-client training-observation sampling (ATC-aware Dirichlet)
│   ├── models/               # Model architectures
│   │   ├── encoder.py        # GraphSAGE GNN encoder with fixed feature projection (Strategy 1)
│   │   └── predictor.py      # Hadamard product MLP predictor
│   ├── training/             # Training & FL logic
│   │   ├── trainer.py        # Local / Centralized Trainer class + build_client_trainer() factory
│   │   ├── evaluate.py       # Hits@20 evaluation functions
│   │   └── fl_client.py      # Flower Client simulation wrapper
│   └── utils/                # General utility functions
│       └── utils.py          # Seeding, device lookup, formatting helpers
├── tests/                    # Unit & Integration test suite (101 tests)
├── colab_setup.py            # Automated Google Colab setup script
├── FedDDI_GNN_Colab.ipynb    # Ready-to-run Colab notebook (full-graph FL + centralized baseline + drift)
├── main_fl.py                # Federated Learning entry point (FedAvg, FedProx, SCAFFOLD, FedDrift)
├── train.py                  # Centralized Baseline entry point (Mode A: learned embeddings, Mode B: fixed features)
├── pyrightconfig.json        # Pyright / Pylance type-checking config
└── .github/workflows/ci.yml  # GitHub Actions CI workflow config
```

## Google Colab Setup Guide

Follow these steps to set up and run the codebase on a Google Colab GPU runtime.
(Go to **Runtime → Change runtime type → T4 GPU** first.)

**Fastest path:** open `FedDDI_GNN_Colab.ipynb` in Colab and run the cells top
to bottom — it does everything below for you, including the full 100-round FL
run, centralized baselines (both modes), and concept drift experiments. The steps
below are the same flow as plain shell cells, if you'd rather build your own notebook.

```python
# 1. Upload and extract the project zip, then enter the folder
#    (in Colab: use the file-browser upload button, or run this cell and
#    pick the zip when prompted)
from google.colab import files
uploaded = files.upload()  # select FedDDI-GNN-step5-concept-drift-fixed.zip

!unzip -oq FedDDI-GNN-step5-concept-drift-fixed.zip
%cd FedDDI-GNN-core

# 2. Install dependencies (compatible with Colab requirements)
!pip install -r requirements.txt

# 3. Run Colab setup check (patches OGB PyTorch compatibility issue)
!python colab_setup.py

# 4. Download the dataset & generate node mapping (first call to
#    load_dataset() triggers the OGB download)
!python -c "from src.data.dataset import load_dataset; load_dataset(root='./data')"

# 5. (Optional) Generate ATC therapeutic class mapping
#    REQUIRED if you want to see therapeutic-area logs in drift / heterogeneous runs
!python scripts/prepare_atc_mapping.py

# 6. Run Centralized Baseline smoke test (learned embeddings, Mode A)
!python train.py training.num_runs=1 training.epochs=50 wandb.mode=disabled device=cuda

# 6b. Run Centralized Baseline smoke test (fixed features, Mode B — MATCHED baseline)
#    This is the ceiling for federated performance.
!python train.py baseline.use_fixed_features=true training.num_runs=1 \
    training.epochs=50 wandb.mode=disabled device=cuda

# 7. Run a quick Federated Learning simulation smoke test
#    (every client trains on the complete shared global graph — see
#    "Aggregation & Graph Sharing" below)
!python main_fl.py \
    num_rounds=2 \
    local_epochs=1 \
    num_clients=2 \
    training.epochs=1 \
    device=cuda \
    wandb.mode=disabled

# 8. Run the full FL simulation (100 rounds, 2 clients, shared global graph, no drift)
#    This is the stable FedAvg baseline — compare against Mode B centralized to measure
#    federation overhead.
!python main_fl.py \
    num_rounds=100 \
    local_epochs=10 \
    num_clients=2 \
    device=cuda \
    wandb.mode=disabled \
    2>&1 | tee fl_baseline.log

# 9. Run FL with concept drift (100 rounds, 2 clients)
#    Client 0 drifts suddenly at round 40; Client 1 drifts gradually starting round 50.
#    The log will show drift/plateau_hits20, drift/nadir_hits20, drift/degradation,
#    and drift/recovery_round metrics.
!python main_fl.py \
    num_rounds=100 \
    local_epochs=10 \
    num_clients=2 \
    device=cuda \
    wandb.mode=disabled \
    'drift.events=[
      {client_id: 0, start_round: 40, kind: sudden,  target_mixture: {0: 0.40, 1: 0.35, 2: 0.25}},
      {client_id: 1, start_round: 50, kind: gradual, transition_rounds: 10, target_mixture: {2: 0.70, 0: 0.30}}
    ]' \
    2>&1 | tee fl_drift.log
```

If you've pushed this code to your own GitHub repo, you can swap step 1 for
`!git clone <your-repo-url>.git` instead of uploading the zip each time.

---

## Local Setup Guide

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Data (ATC Mapping & Node Features)
The dataset must be downloaded once (via `load_dataset`) before
`prepare_atc_mapping.py` can run, since it depends on the node-id mapping
file that OGB creates on first download.

**Note:** `atc_mapping.pt` is not required to run `main_fl.py`'s FedAvg
baseline at all — every client trains on the complete shared graph. Run this
step only if you plan to use the ATC mapping for heterogeneous-observation /
drift experiments, or want therapeutic-area labels in the logs.
```bash
# Automated prepare target (handles the dataset-download dependency for you):
make prepare-data

# Manual alternative — note the required order:
# python -c "from src.data.dataset import load_dataset; load_dataset(root='./data')"
# python scripts/prepare_atc_mapping.py
```


### 3. Run Centralized Baseline

Train the centralized model.

**Mode A (learned embeddings — upper bound, not comparable to FL):**
```bash
# Run quick baseline smoke test (1 run, 50 epochs)
make train-quick

# Run full baseline (5 runs, 500 epochs, leaderboard config)
make train
```

**Mode B (fixed features — matched baseline for FedAvg comparison):**
```bash
# Smoke test
python train.py baseline.use_fixed_features=true training.num_runs=1 \
    training.epochs=50 wandb.mode=disabled device=cuda

# Full run (5 seeds, 500 epochs)
python train.py baseline.use_fixed_features=true training.num_runs=5 \
    training.epochs=500 wandb.mode=disabled device=cuda 2>&1 | tee centralized_fixed_features.log
```

### 4. Run Federated Learning Simulation

Run the virtual Flower simulation. Every client trains on the complete,
shared global DDI graph — this is a clean FedAvg baseline with non-IID
heterogeneous per-client observations.

**Stable baseline (no drift):**
```bash
# FL quick dry-run on CPU (2 rounds, 1 local epoch, 2 clients)
make train-fl-quick

# Full FL simulation on GPU (100 rounds, 10 local epochs, 2 clients, no drift)
python main_fl.py num_rounds=100 local_epochs=10 num_clients=2 \
    device=cuda wandb.mode=disabled 2>&1 | tee fl_baseline.log
```

**With concept drift (research experiments):**
```bash
# 100 rounds, drift at rounds 40 and 50
python main_fl.py num_rounds=100 local_epochs=10 num_clients=2 \
    device=cuda wandb.mode=disabled \
    'drift.events=[
      {client_id: 0, start_round: 40, kind: sudden,  target_mixture: {0: 0.40, 1: 0.35, 2: 0.25}},
      {client_id: 1, start_round: 50, kind: gradual, transition_rounds: 10, target_mixture: {2: 0.70, 0: 0.30}}
    ]' \
    2>&1 | tee fl_drift.log
```

### 5. Code Quality Check
Format and lint the codebase:
```bash
# Reformat with black and autofix ruff lints
make format

# Verify linting and type-safety (Ruff + Pyright)
make lint
```

## Understanding the Output

When you run either the centralized training or the federated simulation, you will see logging in the terminal. Here's how to interpret the results:

### Centralized Baseline Output (`train.py`)
- **`Epoch [X/Y]`**: The current epoch of training over the full graph.
- **`Loss: 0.1234`**: The binary cross-entropy loss computed on the training edges and negatively sampled edges.
- **`Hits@20`**: The primary evaluation metric for OGB-DDI. It measures the ratio of positive validation/test edges that are ranked in the top 20 compared to 100,000 negative edges. A score of `0.8000` means 80% of true interactions were correctly ranked in the top 20.
- **`Run X - Best Val: ..., Test: ...`**: The script reports the peak validation performance and the corresponding test performance at that epoch.
- **W&B Integration**: If Weights & Biases is enabled, you'll get a URL to view the training curves, loss decrease, and performance metrics in real time.

### Federated Hospital Simulation Output (`main_fl.py`)
- **`INFO flwr: Starting Flower simulation`**: Flower initializes the virtual environment for simulation.
- **`Round X - federated train loss: ...`**: weighted-average training loss across clients for that round.
- **`Round X - federated Val Hits@20: ...`**: **this is the primary metric.** Each client evaluates the global validation/test edge set using its own locally-trained node representations combined with the aggregated `feature_proj`/`SAGEConv`/MLP weights, and these per-client scores are weighted-averaged. Because every client trains on the same complete global graph, this should track the server's own evaluation closely.
- **`Round X - server_eval Val Hits@20: ...`**: the server's own global encoder, built with the complete global `edge_index` (train+val) and the just-aggregated weights — the cleanest possible check. This should track the federated number.

**Drift-specific metrics (when drift events are configured):**
- **`Drift plateau frozen at round N: Hits@20 = Y`**: The mean Hits@20 over the 5 stable rounds immediately before the first drift event. This is the pre-drift baseline.
- **`new post-drift nadir: Hits@20 = Z (degradation = X)`**: A new lowest performance seen after drift fires. X = plateau − nadir, the key "how much did drift hurt" metric.
- **`RECOVERED to 95% of plateau (Hits@20 = Z >= Y*0.95). Rounds to recovery: N`**: The model has returned to 95% of pre-drift performance. N shows how many rounds it took.

## Aggregation & Graph Sharing (what's shared, and why)

- **Always aggregated** (Strategy 1, every run): `encoder.feature_proj.*`, `encoder.convs.*` (the `SAGEConv` layers), and `predictor.mlp.*`. There is no private per-client embedding table in this pipeline — every client and the server compute node representations from the same fixed, shared structural feature tensor (`compute_fixed_global_features()`), so FedAvg-averaging these layers is mathematically meaningful (see `main_fl.py` module docstring, "Strategy 1 Fix").
- **Graph structure** (`edge_index` used for `SAGEConv` message-passing): every client uses the **complete** global edge_index — no node or edge is hidden from any client, and there is no client-specific graph construction anywhere in this project. This is a deliberate modeling choice: the old/known `ogbl-ddi` interactions (and the DrugBank-scale catalog this project scales up to) are already-published pharmacological knowledge, not private patient data, so there is no real privacy boundary to enforce by partitioning a public interaction graph across clients.
- **Training edges**: each client trains on its own `LocalObservationDataset` (`src/data/local_observations.py`) — a personalized sample of the global positive training edges drawn via ATC-aware Dirichlet sampling (`dirichlet_alpha`), so different clients are weighted towards different therapeutic areas (e.g. one hospital trains mostly on cardiovascular interactions, another mostly on oncology). This is sampling **with replacement** from the shared global edges, not a partition: clients may share edges, and the message-passing graph is always the complete shared `graph` regardless of which observations a client trains on. **Higher heterogeneity (lower alpha) makes FedAvg harder.**
- **Concept drift** (`src/data/drift.py`): Over a federated training run, selected clients' observation distributions can evolve mid-run (e.g., sudden shift in therapeutic-area mix, or gradual change over rounds). The graph is never modified — only each client's training-observation sampling distribution changes. This simulates real-world hospital dynamics: new drugs entering clinical practice, changing guidelines, shifting patient populations.
- **If you need a real privacy boundary**, it belongs on each hospital's private local data: prescription frequencies, adverse-event observations, and patient outcomes — never on the old, already-public graph.

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model.hidden_dim` | 500 | Embedding + SAGEConv dimension |
| `model.num_layers` | 2 | GNN depth |
| `model.dropout` | 0.3 | Dropout rate |
| `predictor.hidden_dim` | 512 | MLP hidden size |
| `predictor.num_layers` | 3 | MLP depth |
| `training.lr` | 0.003 | Adam LR |
| `training.epochs` | 500 | Max epochs (centralized) |
| `training.batch_size` | 65536 | Edge mini-batch size |
| `local_epochs` | 10 | Local epochs/client/round in FL (critical: was 1, now 10) |
| `local_observations.dirichlet_alpha` | 0.5 | Non-IID heterogeneity level (lower = more skewed, harder for FedAvg) |
| `num_rounds` | 100 | FL rounds (allows pre-drift plateau + post-drift recovery window) |

## Client Setup

Every client receives the identical shared global graph, its own
`GraphSAGEEncoder`/`NeuralLinkPredictor` pair, and its own optimizer. There
is no partitioning step and no per-client subgraph construction — clients
differ only in their local training observations, sampled per-client via
ATC-aware Dirichlet sampling. With `dirichlet_alpha=0.5`, clients specialize
in different therapeutic areas, creating the "honest federation" problem that
FedDrift is designed to solve.