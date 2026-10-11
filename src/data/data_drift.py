"""Simulate client-local feature drift without changing graph topology or edge labels."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from src.data.schedule import ramp_alpha

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftEvent:
    """One client's covariate-drift configuration"""

    client_id: int
    start_round: int
    feature_dims: List[int]
    shift_magnitude: float = 2.0
    noise_std: float = 0.0
    target_node_frac: float = 0.3
    node_selection: str = "high_degree"
    kind: str = "sudden"
    transition_rounds: int = 10


class DataDriftSimulator:
    """Owns per-round, per-client feature-space (covariate) drift"""

    def __init__(
        self,
        node_features: Tensor,
        num_clients: int,
        events: Optional[List[FeatureDriftEvent]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._num_nodes, self._num_feat_dims = node_features.shape
        self._num_clients = num_clients
        self._base_seed = seed

        self._events_by_client: Dict[int, FeatureDriftEvent] = {}
        for ev in events or []:
            if ev.client_id in self._events_by_client:
                logger.warning(
                    "Client %d has more than one FeatureDriftEvent — only "
                    "the last one registered will apply.",
                    ev.client_id,
                )
            self._events_by_client[ev.client_id] = ev





        self._masks: Dict[int, Tensor] = {}
        if self._events_by_client:
            degree_rank = node_features[:, 0]
            for cid, ev in self._events_by_client.items():
                self._masks[cid] = self._select_nodes(ev, degree_rank)
                self._validate_dims(ev)

        if self._events_by_client:
            logger.info(
                "DataDriftSimulator initialised — %d/%d client(s) drift:",
                len(self._events_by_client),
                num_clients,
            )
            for ev in self._events_by_client.values():
                logger.info(
                    "  Client %d | %s drift from round %d | dims=%s "
                    "magnitude=%.2f noise_std=%.2f node_frac=%.2f (%s)",
                    ev.client_id,
                    ev.kind,
                    ev.start_round,
                    ev.feature_dims,
                    ev.shift_magnitude,
                    ev.noise_std,
                    ev.target_node_frac,
                    ev.node_selection,
                )
        else:
            logger.info(
                "DataDriftSimulator initialised — no FeatureDriftEvents "
                "(all %d client(s) stable for the entire run).",
                num_clients,
            )

    def _validate_dims(self, ev: FeatureDriftEvent) -> None:
        bad = [d for d in ev.feature_dims if not (0 <= d < self._num_feat_dims)]
        if bad:
            raise ValueError(
                f"Client {ev.client_id}: feature_dims {bad} out of range "
                f"for a [{self._num_nodes}, {self._num_feat_dims}] feature matrix."
            )

    def _select_nodes(self, ev: FeatureDriftEvent, degree_rank: Tensor) -> Tensor:
        rng = np.random.default_rng(
            None if self._base_seed is None else self._base_seed * 7919 + ev.client_id
        )
        k = max(1, int(round(ev.target_node_frac * self._num_nodes)))
        if ev.node_selection == "high_degree":



            order = torch.argsort(degree_rank, descending=True)
            selected = order[:k]
        elif ev.node_selection == "random":
            selected = torch.from_numpy(
                rng.choice(self._num_nodes, size=k, replace=False)
            ).long()
        else:
            raise ValueError(
                f"Unknown node_selection {ev.node_selection!r} — "
                "expected 'high_degree' or 'random'."
            )
        mask = torch.zeros(self._num_nodes, dtype=torch.bool)
        mask[selected] = True
        return mask

    def _round_rng(self, client_id: int, server_round: int) -> np.random.Generator:
        if self._base_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(
            self._base_seed * 104_729 + client_id * 1009 + server_round
        )

    def step(self, server_round: int) -> Dict[int, Optional[Tensor]]:
        """Return {client_id: additive_delta_or_None} for every client that has an EVENT configured
        (clients with no event are simply absent from the returned dict — callers should treat a
        missing key the same as None, i.e
        """
        result: Dict[int, Optional[Tensor]] = {}
        for cid, ev in self._events_by_client.items():
            severity = ramp_alpha(
                ev.kind, ev.start_round, ev.transition_rounds, server_round
            )
            if severity <= 0.0:
                result[cid] = None
                continue

            delta = torch.zeros(self._num_nodes, self._num_feat_dims)
            mask = self._masks[cid]
            rng = self._round_rng(cid, server_round)

            shift = ev.shift_magnitude * severity
            for dim in ev.feature_dims:
                n_drift = int(mask.sum().item())
                col = torch.full((n_drift,), shift, dtype=torch.float32)
                if ev.noise_std > 0:
                    noise = rng.normal(
                        loc=0.0, scale=ev.noise_std * severity, size=n_drift
                    )
                    col = col + torch.from_numpy(noise).float()
                delta[mask, dim] = col

            result[cid] = delta
        return result

    def drifted_node_mask(self, client_id: int) -> Optional[Tensor]:
        """Bool mask [N], True for nodes whose features drift for this client"""
        return self._masks.get(client_id)
