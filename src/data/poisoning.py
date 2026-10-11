"""Simulate client-local label poisoning and optional parameter-update scaling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from src.data.local_observations import (
    LocalObservationDataset,
    _edge_classes,
    build_edge_lookup,
    sample_observations,
)
from src.data.schedule import ramp_alpha

logger = logging.getLogger(__name__)


def apply_label_flips(
    pos_edges: Tensor,
    train_edges: Tensor,
    atc_mapping: Optional[Tensor],
    suppress: Dict[int, float],
    emerge: Dict[int, float],
    n_obs: int,
    num_nodes: int,
    edge_lookup: frozenset,
    rng: np.random.Generator,
) -> "tuple[Tensor, Tensor, Tensor]":
    """Apply adversarial label flips on top of already-sampled `pos_edges`"""
    pos_edges = pos_edges.clone()
    false_neg_parts: List[np.ndarray] = []

    if suppress:
        edge_classes = _edge_classes(pos_edges, atc_mapping)
        keep_mask = np.ones(pos_edges.size(0), dtype=bool)
        for c, frac in suppress.items():
            if frac <= 0:
                continue
            idx_c = np.where(edge_classes == c)[0]
            if idx_c.size == 0:
                continue
            k = int(round(frac * idx_c.size))
            if k == 0:
                continue
            chosen = rng.choice(idx_c, size=k, replace=False)
            keep_mask[chosen] = False
            false_neg_parts.append(pos_edges[chosen].cpu().numpy())
        pos_edges = pos_edges[torch.from_numpy(keep_mask)]

    false_negative_edges = (
        torch.from_numpy(np.concatenate(false_neg_parts)).long()
        if false_neg_parts
        else torch.empty(0, 2, dtype=torch.long)
    )

    emerged_edges = torch.empty(0, 2, dtype=torch.long)
    if emerge and atc_mapping is not None:
        node_classes = atc_mapping.cpu().numpy()
        emerge_parts: List[np.ndarray] = []
        for c, frac in emerge.items():
            if frac <= 0:
                continue
            k = int(round(frac * n_obs))
            if k == 0:
                continue
            src_pool = np.where(node_classes == c)[0]
            if src_pool.size == 0:
                continue
            found: List[tuple] = []
            attempts = 0
            max_attempts = k * 50 + 1000
            while len(found) < k and attempts < max_attempts:
                attempts += 1
                batch_n = max(k - len(found), 1) * 2
                u = rng.choice(src_pool, size=batch_n)
                v = rng.integers(0, num_nodes, size=batch_n)
                lo = np.minimum(u, v)
                hi = np.maximum(u, v)
                for a, b, x, y in zip(lo, hi, u, v):
                    if x == y:
                        continue
                    if (int(a), int(b)) in edge_lookup:
                        continue
                    found.append((int(x), int(y)))
                    if len(found) >= k:
                        break
            if found:
                emerge_parts.append(np.array(found, dtype=np.int64))
        if emerge_parts:
            emerged_edges = torch.from_numpy(np.concatenate(emerge_parts)).long()
            pos_edges = torch.cat([pos_edges, emerged_edges], dim=0)

    return pos_edges, false_negative_edges, emerged_edges


@dataclass
class PoisonEvent:
    """One malicious client's attack configuration"""

    client_id: int
    start_round: int
    suppress: Dict[int, float] = field(default_factory=dict)
    emerge: Dict[int, float] = field(default_factory=dict)
    kind: str = "sudden"
    transition_rounds: int = 10
    update_scale: float = 1.0


def build_poison_events_from_fraction(
    num_clients: int,
    fraction_malicious: float,
    start_round: int,
    suppress: Dict[int, float],
    emerge: Dict[int, float],
    kind: str = "sudden",
    transition_rounds: int = 10,
    update_scale: float = 1.0,
    seed: Optional[int] = None,
) -> List[PoisonEvent]:
    """Convenience builder: make a deterministic `fraction_malicious` share of clients malicious,
    all sharing the same attack config
    """
    if not (0.0 <= fraction_malicious <= 1.0):
        raise ValueError(f"fraction_malicious must be in [0, 1], got {fraction_malicious}")
    num_malicious = int(round(fraction_malicious * num_clients))
    if num_malicious == 0:
        return []
    rng = np.random.default_rng(seed)
    malicious_ids = rng.choice(num_clients, size=num_malicious, replace=False)
    return [
        PoisonEvent(
            client_id=int(cid),
            start_round=start_round,
            suppress=dict(suppress),
            emerge=dict(emerge),
            kind=kind,
            transition_rounds=transition_rounds,
            update_scale=update_scale,
        )
        for cid in sorted(malicious_ids.tolist())
    ]


class PoisoningSimulator:
    """Owns per-round, per-client data poisoning"""

    def __init__(
        self,
        initial_datasets: List[LocalObservationDataset],
        train_edges: Tensor,
        atc_mapping: Optional[Tensor],
        events: Optional[List[PoisonEvent]] = None,
        observations_per_client: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not initial_datasets:
            raise ValueError("initial_datasets must not be empty")

        self._train_edges = train_edges
        self._atc_mapping = atc_mapping
        self._num_nodes = int(train_edges.max().item()) + 1
        self._n_obs = observations_per_client or len(initial_datasets[0])
        self._base_seed = seed
        self._num_clients = len(initial_datasets)

        self._clean_mixtures: List[Dict[int, float]] = [
            dict(ds.class_mixture) if ds.class_mixture else {0: 1.0}
            for ds in initial_datasets
        ]

        self._events_by_client: Dict[int, PoisonEvent] = {}
        for ev in events or []:
            if ev.client_id in self._events_by_client:
                logger.warning(
                    "Client %d has more than one PoisonEvent — only the "
                    "last one registered will apply.",
                    ev.client_id,
                )
            self._events_by_client[ev.client_id] = ev

        self._edge_lookup = (
            build_edge_lookup(train_edges) if self._events_by_client else frozenset()
        )
        if self._events_by_client and atc_mapping is None:
            logger.warning(
                "PoisonEvent(s) configured but no ATC mapping was provided — "
                "`emerge` will have no effect without one (`suppress` still works)."
            )

        if self._events_by_client:
            logger.info(
                "PoisoningSimulator initialised — %d/%d client(s) malicious:",
                len(self._events_by_client),
                self._num_clients,
            )
            for ev in self._events_by_client.values():
                logger.info(
                    "  Client %d | %s attack from round %d | suppress=%s emerge=%s "
                    "update_scale=%.2f",
                    ev.client_id,
                    ev.kind,
                    ev.start_round,
                    ev.suppress,
                    ev.emerge,
                    ev.update_scale,
                )
        else:
            logger.info(
                "PoisoningSimulator initialised — no PoisonEvents (all %d "
                "client(s) honest for the entire run).",
                self._num_clients,
            )

    def _round_rng(self, server_round: int) -> np.random.Generator:
        if self._base_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self._base_seed * 1_000_003 + server_round)

    def step(self, server_round: int) -> List[LocalObservationDataset]:
        """Return this round's dataset for every client"""
        rng = self._round_rng(server_round)
        datasets: List[LocalObservationDataset] = []

        for cid in range(self._num_clients):
            base_edges = sample_observations(
                self._train_edges,
                self._atc_mapping,
                self._clean_mixtures[cid],
                self._n_obs,
                rng,
            )

            event = self._events_by_client.get(cid)
            if event is None or server_round < event.start_round:
                datasets.append(
                    LocalObservationDataset(
                        client_id=cid,
                        edges=base_edges,
                        class_mixture=self._clean_mixtures[cid],
                    )
                )
                continue

            severity = ramp_alpha(
                event.kind, event.start_round, event.transition_rounds, server_round
            )
            suppress = {c: frac * severity for c, frac in event.suppress.items()}
            emerge = {c: frac * severity for c, frac in event.emerge.items()}

            pos_edges, false_neg, emerged = apply_label_flips(
                pos_edges=base_edges,
                train_edges=self._train_edges,
                atc_mapping=self._atc_mapping,
                suppress=suppress,
                emerge=emerge,
                n_obs=self._n_obs,
                num_nodes=self._num_nodes,
                edge_lookup=self._edge_lookup,
                rng=rng,
            )
            datasets.append(
                LocalObservationDataset(
                    client_id=cid,
                    edges=pos_edges,
                    class_mixture=self._clean_mixtures[cid],
                    false_negative_edges=false_neg,
                    emerged_edges=emerged,
                )
            )

        return datasets

    def update_scale(self, client_id: int, server_round: int) -> float:
        """Return the parameter-delta scaling factor a malicious client should apply this round
        (1.0 for honest clients / inactive attacks)
        """
        event = self._events_by_client.get(client_id)
        if event is None or server_round < event.start_round:
            return 1.0
        return float(event.update_scale)

    def is_malicious(self, client_id: int) -> bool:
        return client_id in self._events_by_client
