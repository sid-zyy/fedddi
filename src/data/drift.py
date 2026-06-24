"""
src/data/drift.py
──────────────────────────────────────────────────────────────────────────────
Step 5 — Concept Drift Simulator for FedDrift.

Design contract
───────────────
The graph NEVER changes. Only each client's LOCAL OBSERVATION DISTRIBUTION
changes over time. This matches the FedDrift research proposal exactly:

    Hospital A (Round 0): Cardiovascular 70% / Diabetes 20% / Oncology 10%
    Hospital A (Round 50): Cardiovascular 40% / Oncology 35% / Diabetes 25%
    Hospital B: unchanged throughout.

The simulator works by storing a TARGET MIXTURE per client that it updates
each round according to a drift schedule. It then calls `sample_observations()`
from `local_observations.py` — the SAME primitive introduced in Step 4 — to
re-sample that client's local observations according to the new mixture. No
new sampling logic is introduced here.

Key invariants
──────────────
  • Every `LocalObservationDataset` returned by `step()` is sampled from the
    same global `train_edges` tensor — never a subset of nodes.
  • The message-passing graph passed to every client is NEVER touched by this
    module. It has no reference to any encoder, graph, or trainer.
  • Two clients may legally share edges in their observation sets; that is how
    prescription FREQUENCY rather than exclusivity is modelled.

Drift schedule types
────────────────────
  sudden   — The mixture jumps from the initial to the target in one round.
             Useful for simulating a new drug becoming available or a policy
             change that takes effect immediately.

  gradual  — The mixture transitions linearly from the initial to the target
             over `transition_rounds` rounds, reaching the target smoothly.
             Useful for demographic shifts (e.g., an ageing patient base) that
             accumulate over months.

Multi-client, multi-drift support
──────────────────────────────────
Each `DriftEvent` attaches to exactly one client and fires at one `start_round`.
Multiple events may target the same client (e.g., two successive sudden shifts)
or different clients. Clients with no events keep their initial mixture
unchanged for the whole run — they are "stable" hospitals in the FedDrift
terminology.

Integration with main_fl.py
────────────────────────────
`ConceptDriftSimulator` is constructed once before the Flower simulation loop
and is called inside the `client_fn` or a per-round hook. The typical pattern:

    simulator = ConceptDriftSimulator(
        initial_datasets=client_datasets,    # from generate_client_observations()
        train_edges=split["train"]["edge"],
        atc_mapping=atc_mapping,
        events=[
            DriftEvent(
                client_id=0,
                start_round=25,
                target_mixture={0: 0.40, 1: 0.35, 2: 0.25},
                kind="sudden",
            ),
        ],
        seed=cfg.training.seed,
    )

    # Inside a per-round update hook:
    current_datasets = simulator.step(server_round)
    # Each client's trainer's train split is updated to current_datasets[cid].edges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from src.data.local_observations import (
    LocalObservationDataset,
    sample_observations,
)

logger = logging.getLogger(__name__)

DriftKind = Literal["sudden", "gradual"]


@dataclass
class DriftEvent:
    """Specification for one concept-drift event affecting one client.

    Parameters
    ----------
    client_id : int
        Which client experiences this drift.
    start_round : int
        The first FL round at which this event activates.
        For ``kind="sudden"`` the target mixture is applied immediately
        at this round (inclusive).
        For ``kind="gradual"`` the transition BEGINS at this round and
        completes at ``start_round + transition_rounds - 1``.
    target_mixture : Dict[int, float]
        The desired per-ATC-class proportions AFTER the drift. Values
        need not sum to 1 — they are renormalised internally. Classes
        absent from this dict are implicitly assigned weight 0.
        Note: if the global ``train_edges`` has no edges for a given
        class, that class is silently dropped (same behaviour as
        ``sample_observations()`` in local_observations.py).
    kind : {"sudden", "gradual"}
        Drift schedule type. Default: ``"sudden"``.
    transition_rounds : int
        Number of rounds over which a ``"gradual"`` drift linearly
        interpolates from the initial to the target mixture. Ignored for
        ``"sudden"`` drift. Must be >= 1 when kind="gradual".
    """

    client_id: int
    start_round: int
    target_mixture: Dict[int, float]
    kind: DriftKind = "sudden"
    transition_rounds: int = 10

    def __post_init__(self) -> None:
        if self.kind not in ("sudden", "gradual"):
            raise ValueError(
                f"DriftEvent.kind must be 'sudden' or 'gradual', got {self.kind!r}"
            )
        if self.kind == "gradual" and self.transition_rounds < 1:
            raise ValueError(
                f"transition_rounds must be >= 1 for gradual drift, "
                f"got {self.transition_rounds}"
            )
        if not self.target_mixture:
            raise ValueError("target_mixture must not be empty")
        if any(p < 0 for p in self.target_mixture.values()):
            raise ValueError("target_mixture values must all be >= 0")


def _normalise(mixture: Dict[int, float]) -> Dict[int, float]:
    """Return a copy of `mixture` with values normalised to sum to 1."""
    total = sum(mixture.values())
    if total <= 0:
        raise ValueError("mixture has no positive weight — cannot normalise")
    return {c: p / total for c, p in mixture.items()}


def _interpolate(
    start: Dict[int, float],
    end: Dict[int, float],
    alpha: float,
) -> Dict[int, float]:
    """Linear interpolation between two mixtures.

    alpha=0.0  → start mixture
    alpha=1.0  → end mixture

    Classes present in only one of the two dicts get weight 0 in the
    other for the purposes of interpolation, then the result is
    renormalised so it always sums to 1.
    """
    all_classes = set(start) | set(end)
    interp = {
        c: (1.0 - alpha) * start.get(c, 0.0) + alpha * end.get(c, 0.0)
        for c in all_classes
    }
    return _normalise(interp)


class ConceptDriftSimulator:
    """Manages evolving local observation distributions across FL rounds.

    The simulator owns the CURRENT per-client mixture and re-samples
    each client's ``LocalObservationDataset`` every time ``step()`` is
    called. Clients with no active drift event keep their initial mixture
    unchanged — ``step()`` still re-samples their observations (with the
    same mixture) so that the returned list of datasets is always fresh
    and independently shuffled, even for stable clients.

    Parameters
    ----------
    initial_datasets : List[LocalObservationDataset]
        One dataset per client from ``generate_client_observations()``.
        Their ``class_mixture`` fields are used as the starting point for
        any drift event targeting that client.
    train_edges : Tensor [E, 2]
        The shared global training edge tensor. Never partitioned.
    atc_mapping : Optional Tensor [num_nodes]
        ATC Level-1 class index per node (same as in local_observations.py).
        If None, class-agnostic sampling is used (all edges are one class).
    events : List[DriftEvent]
        Drift events to simulate. Multiple events may target the same
        client and will fire in ``start_round`` order. Events that overlap
        in time on the same client use the last one to fire as the
        authoritative ``start_mixture`` for subsequent interpolation.
    observations_per_client : Optional[int]
        Number of observations to sample per client per step. Defaults to
        ``len(initial_datasets[0])`` (i.e., matches the Step-4 default).
    seed : Optional[int]
        Base RNG seed. The simulator derives per-round sub-seeds
        deterministically so that ``step(r)`` is always reproducible
        given the same seed and the same active mixture at round ``r``.
    """

    def __init__(
        self,
        initial_datasets: List[LocalObservationDataset],
        train_edges: Tensor,
        atc_mapping: Optional[Tensor],
        events: Optional[List[DriftEvent]] = None,
        observations_per_client: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not initial_datasets:
            raise ValueError("initial_datasets must not be empty")

        self._train_edges = train_edges
        self._atc_mapping = atc_mapping
        self._n_obs = observations_per_client or len(initial_datasets[0])
        self._base_seed = seed

        # Per-client current mixture (starts at each dataset's Dirichlet draw).
        self._current_mixtures: List[Dict[int, float]] = [
            _normalise(ds.class_mixture) if ds.class_mixture else {0: 1.0}
            for ds in initial_datasets
        ]

        # Events sorted by start_round so earlier events fire before later ones.
        self._events: List[DriftEvent] = sorted(
            events or [], key=lambda e: e.start_round
        )

        self._num_clients = len(initial_datasets)
        self._last_step_round: Optional[int] = None

        # For gradual drift: track the start mixture AT THE TIME the event fires
        # (not the initial mixture) so chained gradual events work correctly.
        # Keyed by (client_id, event_index_in_self._events).
        self._gradual_start_mixtures: Dict[tuple, Dict[int, float]] = {}

        # Log summary of configured events.
        if self._events:
            logger.info(
                "ConceptDriftSimulator initialised — %d client(s), %d drift event(s):",
                self._num_clients,
                len(self._events),
            )
            for ev in self._events:
                if ev.kind == "sudden":
                    logger.info(
                        "  Client %d | sudden drift at round %d → %s",
                        ev.client_id,
                        ev.start_round,
                        {c: f"{p:.0%}" for c, p in ev.target_mixture.items()},
                    )
                else:
                    logger.info(
                        "  Client %d | gradual drift starting round %d "
                        "(over %d rounds) → %s",
                        ev.client_id,
                        ev.start_round,
                        ev.transition_rounds,
                        {c: f"{p:.0%}" for c, p in ev.target_mixture.items()},
                    )
        else:
            logger.info(
                "ConceptDriftSimulator initialised — %d client(s), no drift events "
                "(stable distributions for all clients).",
                self._num_clients,
            )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _effective_mixture(
        self, client_id: int, server_round: int
    ) -> Dict[int, float]:
        """Return the effective mixture for `client_id` at `server_round`.

        Processes events in start_round order; each event that applies at
        this round overrides or blends into the mixture computed so far.
        This means chained events compose correctly: a second sudden drift
        after a first one replaces the first's target, not the initial.
        """
        mixture = dict(self._current_mixtures[client_id])

        for ev_idx, ev in enumerate(self._events):
            if ev.client_id != client_id:
                continue
            if server_round < ev.start_round:
                # Event hasn't fired yet.
                continue

            key = (client_id, ev_idx)

            if ev.kind == "sudden":
                mixture = _normalise(ev.target_mixture)

            else:  # "gradual"
                # Capture the mixture at the MOMENT this event first became
                # active (so that chained gradual events don't re-derive from
                # the original initial mixture).
                if key not in self._gradual_start_mixtures:
                    self._gradual_start_mixtures[key] = dict(mixture)

                start_mix = self._gradual_start_mixtures[key]
                rounds_elapsed = server_round - ev.start_round
                alpha = min(1.0, rounds_elapsed / max(ev.transition_rounds - 1, 1))
                mixture = _interpolate(start_mix, ev.target_mixture, alpha)

        return mixture

    def _rng_for_round(self, server_round: int) -> np.random.Generator:
        """Deterministic per-round RNG derived from the base seed."""
        if self._base_seed is None:
            return np.random.default_rng()
        # Mix round into seed to get independent, reproducible draws per round.
        return np.random.default_rng(self._base_seed * 100_000 + server_round)

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, server_round: int) -> List[LocalObservationDataset]:
        """Advance the simulation to `server_round` and return fresh datasets.

        For every client:
          1. Compute the effective mixture at this round (accounting for any
             drift events that have fired).
          2. Update the stored current mixture (so ``current_mixtures`` can be
             inspected externally at any point).
          3. Re-sample ``n_obs`` observations from ``train_edges`` using
             ``sample_observations()`` — the same low-level primitive from
             Step 4 — with the effective mixture.
          4. Return a new ``LocalObservationDataset`` with the fresh edges.

        The graph (``train_edges`` node set / topology) is NEVER modified.

        Parameters
        ----------
        server_round : int
            Current FL round number (1-indexed, matching Flower's convention).
            ``step()`` is idempotent if called multiple times with the same
            round number.

        Returns
        -------
        List[LocalObservationDataset]
            One fresh dataset per client, index == client_id.
        """
        rng = self._rng_for_round(server_round)
        datasets: List[LocalObservationDataset] = []

        for cid in range(self._num_clients):
            new_mixture = self._effective_mixture(cid, server_round)
            self._current_mixtures[cid] = new_mixture

            edges = sample_observations(
                self._train_edges,
                self._atc_mapping,
                new_mixture,
                self._n_obs,
                rng,
            )

            ds = LocalObservationDataset(
                client_id=cid,
                edges=edges,
                class_mixture=new_mixture,
            )
            datasets.append(ds)

        if server_round != self._last_step_round:
            drifting = [
                cid
                for cid in range(self._num_clients)
                if any(
                    ev.client_id == cid and ev.start_round <= server_round
                    for ev in self._events
                )
            ]
            if drifting:
                for cid in drifting:
                    top = sorted(
                        self._current_mixtures[cid].items(), key=lambda kv: -kv[1]
                    )[:3]
                    top_str = ", ".join(f"class {c}: {p:.0%}" for c, p in top)
                    logger.info(
                        "[Round %d] Client %d drift → [%s]",
                        server_round,
                        cid,
                        top_str,
                    )
            self._last_step_round = server_round

        return datasets

    # ── inspection helpers ────────────────────────────────────────────────────

    @property
    def current_mixtures(self) -> List[Dict[int, float]]:
        """Return the current per-client mixtures (read-only snapshot)."""
        return [dict(m) for m in self._current_mixtures]

    def mixture_at(self, client_id: int, server_round: int) -> Dict[int, float]:
        """Return the effective mixture for `client_id` at `server_round`
        WITHOUT mutating simulator state. Useful for unit tests and analysis.
        """
        if not (0 <= client_id < self._num_clients):
            raise ValueError(
                f"client_id {client_id} out of range [0, {self._num_clients})"
            )
        return self._effective_mixture(client_id, server_round)
