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
    """One federated client's private local clinical observations"""

    client_id: int
    edges: Tensor
    class_mixture: Dict[int, float] = field(default_factory=dict)
    false_negative_edges: Tensor = field(
        default_factory=lambda: torch.empty(0, 2, dtype=torch.long)
    )
    """POISONING overlay (see `src/data/poisoning.py:apply_label_flips`).
    Edges that ARE true positive interactions in the global graph, but
    that this (malicious) client's local training LABELS AS NEGATIVE
    (label=0) — e.g. a compromised client suppressing a known
    interaction. Distinct from `edges` (label=1): a member of
    `false_negative_edges` asserts something FALSE about a fixed edge,
    which is what makes this an adversarial corruption rather than a
    change in observation frequency or an honest covariate shift. Empty
    (the default) for honest clients / rounds with no active attack.
    """
    emerged_edges: Tensor = field(
        default_factory=lambda: torch.empty(0, 2, dtype=torch.long)
    )
    """POISONING overlay, counterpart to `false_negative_edges`. Synthetic
    edges injected by `apply_label_flips` (via `emerge`) that are NOT true
    interactions in the global graph, but that this (malicious) client's
    local training LABELS AS POSITIVE (label=1) — e.g. a compromised
    client fabricating a new, unverified interaction. These edges are
    ALSO present (mixed in, indistinguishable) inside `edges` — they are
    trained on exactly like any other positive observation. This field
    exists purely so a caller can identify and directly diagnose these
    specific edges later (see `score_edge_set` in
    `src/training/evaluate.py`, used to measure poisoning ATTACK SUCCESS —
    does the aggregated global model actually believe the fabrication?),
    separately from tracking what fraction of `edges` they represent.
    Empty (the default) for honest clients / rounds with no active attack.
    """

    def __len__(self) -> int:
        return int(self.edges.size(0))

    def __repr__(self) -> str:
        top = sorted(self.class_mixture.items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"class {c}: {p:.0%}" for c, p in top)
        flip_str = (
            f", false_neg={self.false_negative_edges.size(0)}"
            if self.false_negative_edges.size(0) > 0
            else ""
        )
        emerge_str = (
            f", emerged={self.emerged_edges.size(0)}"
            if self.emerged_edges.size(0) > 0
            else ""
        )
        return (
            f"LocalObservationDataset(client={self.client_id}, "
            f"n_observations={len(self)}, top_classes=[{top_str}]"
            f"{flip_str}{emerge_str})"
        )


def _edge_classes(train_edges: Tensor, atc_mapping: Optional[Tensor]) -> np.ndarray:
    """Map every training edge to a therapeutic-area class using its SOURCE node's ATC Level-1 code
    (see scripts/prepare_atc_mapping.py)
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
    """Sample `n_obs` training edges (WITH replacement) according to an explicit per-ATC-class
    target `mixture`
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


def build_edge_lookup(train_edges: Tensor) -> frozenset:
    """Build an O(1)-membership set of undirected edges for negative rejection-sampling in
    `src/data/poisoning.py:apply_label_flips`
    """
    arr = train_edges.cpu().numpy()

    lo = np.minimum(arr[:, 0], arr[:, 1])
    hi = np.maximum(arr[:, 0], arr[:, 1])
    return frozenset(zip(lo.tolist(), hi.tolist()))


def generate_client_observations(
    train_edges: Tensor,
    num_clients: int,
    atc_mapping: Optional[Tensor] = None,
    observations_per_client: Optional[int] = None,
    dirichlet_alpha: float = 0.5,
    seed: Optional[int] = None,
) -> List[LocalObservationDataset]:
    """Generate one heterogeneous `LocalObservationDataset` per client"""
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
