"""
src/data/local_observations.py
────────────────────────────────
Heterogeneous LOCAL OBSERVATIONS for federated clients (FedDrift direction).

Per the FedDrift research proposal: hospitals do not own different subsets
of the DDI GRAPH — drug knowledge is already public (DrugBank, ChEMBL,
OGB-DDI are globally accessible to every institution). Instead, hospitals
own distinct CLINICAL REALITIES: different mixes of which interactions they
observe/prescribe, reflecting their patient population and therapeutic-area
specialization.

This module partitions / samples TRAINING OBSERVATIONS (edges used as
positive training targets) — it never partitions nodes and never touches
the message-passing graph itself. Every client still trains GraphSAGE
message-passing over the complete shared global graph (see
`build_client_trainer` in `src/training/trainer.py` and the "Graph
ownership model" docstring in `main_fl.py`) — only each client's local
training TARGETS differ.

Heterogeneity is generated via ATC-aware Dirichlet sampling, matching the
proposal's worked example almost exactly:

    Hospital A: 70% Cardiovascular / 20% Diabetes / 10% Oncology
    Hospital B: 65% Oncology / 20% Antibiotics / 15% Cardiovascular
    Hospital C: 60% Infectious Disease / 25% Respiratory / 15% Cardiovascular

Each client draws a personalized mixture over ATC Level-1 therapeutic
classes from a symmetric Dirichlet(alpha) distribution, then samples its
local observations from the global training edges according to that
mixture — WITH replacement. Repeatedly "observing" / prescribing the same
already-known interaction is realistic and is how prescription FREQUENCY
differs across hospitals; it is not edge ownership, and two clients may
legitimately share edges in their local observation sets.

`sample_observations()` is the shared low-level primitive (given an
explicit target mixture, draw N observations) — `generate_client_observations()`
is the higher-level entry point that draws Dirichlet-random mixtures for
every client. A later concept-drift step reuses `sample_observations()`
directly with a new TARGET mixture per round, without changing this
module's contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class LocalObservationDataset:
    """One federated client's private local clinical observations.

    This is NOT a graph partition. `edges` is a (possibly repeating)
    multiset of training interactions drawn from the SAME global graph
    every other client also trains GraphSAGE message-passing over. Two
    clients may legitimately share edges — what differs between clients is
    the relative FREQUENCY/MIX with which each one observes interactions
    from different therapeutic areas, not which edges they "own".

    Attributes
    ----------
    client_id : int
    edges : LongTensor [num_observations, 2]
        This client's local positive training observations (src, dst
        pairs). May contain duplicates by design (see module docstring).
    class_mixture : Dict[int, float]
        The ATC Level-1 class proportions this client's observations were
        sampled from. Kept for inspection/logging and reused as the
        starting point for round-by-round concept-drift simulation.
    """

    client_id: int
    edges: Tensor
    class_mixture: Dict[int, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.edges.size(0))

    def __repr__(self) -> str:
        top = sorted(self.class_mixture.items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"class {c}: {p:.0%}" for c, p in top)
        return (
            f"LocalObservationDataset(client={self.client_id}, "
            f"n_observations={len(self)}, top_classes=[{top_str}])"
        )


def _edge_classes(train_edges: Tensor, atc_mapping: Optional[Tensor]) -> np.ndarray:
    """Map every training edge to a therapeutic-area class using its SOURCE
    node's ATC Level-1 code (see scripts/prepare_atc_mapping.py).

    If no ATC mapping is available, every edge is treated as belonging to
    one single class — heterogeneity then comes only from which specific
    edges / how often each client samples them, not from a therapeutic
    mix. This lets the rest of the pipeline keep working (e.g. in tests,
    or before `prepare_atc_mapping.py` has been run) without crashing.
    """
    if atc_mapping is None:
        return np.zeros(train_edges.size(0), dtype=np.int64)
    src_nodes = train_edges[:, 0].cpu().numpy()
    return atc_mapping.cpu().numpy()[src_nodes]


def sample_observations(
    train_edges: Tensor,
    atc_mapping: Optional[Tensor],
    mixture: Dict[int, float],
    n_obs: int,
    rng: np.random.Generator,
) -> Tensor:
    """Sample `n_obs` training edges (WITH replacement) according to an
    explicit per-ATC-class target `mixture`.

    This is the shared low-level primitive: `generate_client_observations`
    below calls it with a Dirichlet-random mixture per client; a later
    concept-drift simulator calls it directly with an explicit, evolving
    TARGET mixture per round, on the same global `train_edges`.

    Classes in `mixture` that have zero matching edges in `train_edges`
    are dropped and the remaining probabilities renormalized, so this
    never crashes on an empty pool — it just can't honor a class that
    truly has no representative edges yet.
    """
    edge_classes = _edge_classes(train_edges, atc_mapping)
    classes_present = np.unique(edge_classes)
    class_to_edge_idx = {c: np.where(edge_classes == c)[0] for c in classes_present}

    usable = {c: p for c, p in mixture.items() if c in class_to_edge_idx and p > 0}
    if not usable:
        raise ValueError(
            "None of the requested mixture's ATC classes have any matching "
            "training edges — cannot sample observations."
        )
    total = sum(usable.values())
    usable = {c: p / total for c, p in usable.items()}

    classes = list(usable.keys())
    probs = np.array([usable[c] for c in classes])
    counts = rng.multinomial(n_obs, probs)

    parts: List[np.ndarray] = []
    for c, k in zip(classes, counts):
        if k == 0:
            continue
        pool = class_to_edge_idx[c]
        parts.append(rng.choice(pool, size=int(k), replace=True))

    sampled_idx = (
        np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    )
    rng.shuffle(sampled_idx)
    return train_edges[torch.from_numpy(sampled_idx).long()]


def generate_client_observations(
    train_edges: Tensor,
    num_clients: int,
    atc_mapping: Optional[Tensor] = None,
    observations_per_client: Optional[int] = None,
    dirichlet_alpha: float = 0.5,
    seed: Optional[int] = None,
) -> List[LocalObservationDataset]:
    """Generate one heterogeneous `LocalObservationDataset` per client.

    Does NOT touch the graph or its nodes: `train_edges` is sampled FROM
    (with replacement), never restricted to a node-induced subgraph. Every
    edge returned for every client is a real edge from the shared global
    graph's training split.

    Parameters
    ----------
    train_edges : LongTensor [E, 2]
        The GLOBAL positive training edges (identical for every client's
        message-passing graph).
    num_clients : int
    atc_mapping : Optional LongTensor [num_nodes]
        Maps node -> ATC Level-1 class index (see
        scripts/prepare_atc_mapping.py). If None, heterogeneity is
        class-agnostic (every edge is one class).
    observations_per_client : Optional[int]
        How many local observations to sample per client. Defaults to
        `len(train_edges)` — each client trains on as many local
        observations as the original full training set, just resampled
        according to its own mixture.
    dirichlet_alpha : float
        Concentration parameter for each client's personal class mixture.
        Smaller alpha => more skewed/heterogeneous mixtures across clients
        (alpha -> inf => every client converges to the same, uniform
        mixture over classes). Matches the standard non-IID FL Dirichlet
        partitioning convention (Hsu et al., 2019) — applied here to
        OBSERVATIONS, not to nodes.
    seed : Optional[int]
        RNG seed. None => a fresh, unseeded default RNG.

    Returns
    -------
    List[LocalObservationDataset], one per client, index == client_id.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if dirichlet_alpha <= 0:
        raise ValueError(f"dirichlet_alpha must be > 0, got {dirichlet_alpha}")

    rng = np.random.default_rng(seed)

    edge_classes = _edge_classes(train_edges, atc_mapping)
    classes = np.unique(edge_classes)
    num_classes = len(classes)
    n_obs = observations_per_client or int(train_edges.size(0))

    datasets: List[LocalObservationDataset] = []
    for cid in range(num_clients):
        mixture_vec = rng.dirichlet(np.full(num_classes, dirichlet_alpha))
        mixture = {int(c): float(p) for c, p in zip(classes, mixture_vec)}

        edges = sample_observations(train_edges, atc_mapping, mixture, n_obs, rng)

        ds = LocalObservationDataset(
            client_id=cid, edges=edges, class_mixture=mixture
        )
        datasets.append(ds)
        logger.info("[Client %d] local observations: %s", cid, ds)

    return datasets
