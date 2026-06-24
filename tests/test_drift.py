# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""
tests/test_drift.py
────────────────────
Tests for Step 5: concept drift simulation (src/data/drift.py).

Verifies:
  • Stable clients keep their initial mixture throughout.
  • Sudden drift applies the target mixture immediately at start_round and
    holds it afterwards.
  • Gradual drift interpolates linearly from the initial to the target
    mixture over the specified transition_rounds window, then holds the
    target.
  • Multiple drift events on the same client compose correctly (chaining).
  • Multiple clients can drift independently.
  • The graph is NEVER touched — every observation edge belongs to the
    global train_edges; no new edges are invented.
  • step() is reproducible under a fixed seed.
  • mixture_at() returns correct values without mutating simulator state.
  • Invalid DriftEvent parameters are rejected at construction.
  • Simulator with no events behaves like a stable FedAvg round.
"""

from __future__ import annotations

import pytest
import numpy as np
import torch

from src.data.drift import ConceptDriftSimulator, DriftEvent, _interpolate, _normalise
from src.data.local_observations import (
    LocalObservationDataset,
    generate_client_observations,
)

# ── Shared synthetic fixtures ─────────────────────────────────────────────────

NUM_NODES = 60
NUM_EDGES = 400
NUM_ATC_CLASSES = 4
NUM_CLIENTS = 3


@pytest.fixture
def train_edges() -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    src = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    return torch.stack([src, dst], dim=1)


@pytest.fixture
def atc_mapping() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, NUM_ATC_CLASSES, (NUM_NODES,), generator=g)


@pytest.fixture
def initial_datasets(train_edges, atc_mapping) -> list[LocalObservationDataset]:
    return generate_client_observations(
        train_edges,
        num_clients=NUM_CLIENTS,
        atc_mapping=atc_mapping,
        observations_per_client=100,
        dirichlet_alpha=0.5,
        seed=0,
    )


def _edge_set(edges: torch.Tensor) -> set:
    return {tuple(e) for e in edges.tolist()}


# ── Helper: build a simulator quickly ────────────────────────────────────────

def _make_sim(
    initial_datasets,
    train_edges,
    atc_mapping,
    events=None,
    seed=0,
) -> ConceptDriftSimulator:
    return ConceptDriftSimulator(
        initial_datasets=initial_datasets,
        train_edges=train_edges,
        atc_mapping=atc_mapping,
        events=events or [],
        observations_per_client=100,
        seed=seed,
    )


# ── Unit tests for internal helpers ──────────────────────────────────────────

class TestNormalise:
    def test_sums_to_one(self):
        m = _normalise({0: 2.0, 1: 6.0, 2: 2.0})
        assert abs(sum(m.values()) - 1.0) < 1e-9

    def test_ratios_preserved(self):
        m = _normalise({0: 1.0, 1: 3.0})
        assert abs(m[0] - 0.25) < 1e-9
        assert abs(m[1] - 0.75) < 1e-9

    def test_raises_on_zero_weight(self):
        with pytest.raises(ValueError):
            _normalise({0: 0.0, 1: 0.0})


class TestInterpolate:
    def test_alpha_zero_is_start(self):
        start = {0: 0.7, 1: 0.3}
        end = {0: 0.2, 1: 0.8}
        result = _interpolate(start, end, 0.0)
        assert abs(result[0] - 0.7) < 1e-9

    def test_alpha_one_is_end(self):
        start = {0: 0.7, 1: 0.3}
        end = {0: 0.2, 1: 0.8}
        result = _interpolate(start, end, 1.0)
        assert abs(result[0] - 0.2) < 1e-9

    def test_midpoint(self):
        start = {0: 1.0, 1: 0.0}
        end = {0: 0.0, 1: 1.0}
        result = _interpolate(start, end, 0.5)
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9

    def test_result_sums_to_one(self):
        start = {0: 0.6, 1: 0.4}
        end = {0: 0.1, 2: 0.9}
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _interpolate(start, end, alpha)
            assert abs(sum(result.values()) - 1.0) < 1e-9


# ── DriftEvent validation ─────────────────────────────────────────────────────

class TestDriftEventValidation:
    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            DriftEvent(
                client_id=0,
                start_round=5,
                target_mixture={0: 1.0},
                kind="linear",  # type: ignore[arg-type]
            )

    def test_gradual_zero_transition_rounds_raises(self):
        with pytest.raises(ValueError, match="transition_rounds"):
            DriftEvent(
                client_id=0,
                start_round=5,
                target_mixture={0: 1.0},
                kind="gradual",
                transition_rounds=0,
            )

    def test_empty_target_mixture_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DriftEvent(client_id=0, start_round=5, target_mixture={})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            DriftEvent(client_id=0, start_round=5, target_mixture={0: -0.1, 1: 1.0})

    def test_valid_sudden_event_constructs(self):
        ev = DriftEvent(client_id=1, start_round=10, target_mixture={0: 0.5, 1: 0.5})
        assert ev.kind == "sudden"

    def test_valid_gradual_event_constructs(self):
        ev = DriftEvent(
            client_id=2,
            start_round=20,
            target_mixture={2: 1.0},
            kind="gradual",
            transition_rounds=5,
        )
        assert ev.kind == "gradual"


# ── No-drift baseline ─────────────────────────────────────────────────────────

class TestNoDrift:
    def test_returns_one_dataset_per_client(self, initial_datasets, train_edges, atc_mapping):
        sim = _make_sim(initial_datasets, train_edges, atc_mapping, events=[])
        result = sim.step(1)
        assert len(result) == NUM_CLIENTS
        assert all(isinstance(d, LocalObservationDataset) for d in result)

    def test_every_observation_is_a_real_global_edge(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = _make_sim(initial_datasets, train_edges, atc_mapping)
        for rnd in [1, 10, 50]:
            datasets = sim.step(rnd)
            global_edges = _edge_set(train_edges)
            for ds in datasets:
                assert _edge_set(ds.edges).issubset(global_edges)

    def test_stable_clients_keep_initial_mixture(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = _make_sim(initial_datasets, train_edges, atc_mapping)
        for rnd in [1, 25, 50]:
            for cid, ds in enumerate(sim.step(rnd)):
                init_mix = initial_datasets[cid].class_mixture
                for cls, init_p in init_mix.items():
                    assert abs(ds.class_mixture.get(cls, 0.0) - init_p / sum(init_mix.values())) < 1e-9

    def test_graph_is_never_touched(self, initial_datasets, train_edges, atc_mapping):
        """Simulator must not mutate train_edges or return edges outside it."""
        original_edges = train_edges.clone()
        sim = _make_sim(initial_datasets, train_edges, atc_mapping)
        sim.step(1)
        sim.step(25)
        sim.step(50)
        assert torch.equal(train_edges, original_edges)


# ── Sudden drift ──────────────────────────────────────────────────────────────

class TestSuddenDrift:
    def _make_sudden_sim(self, initial_datasets, train_edges, atc_mapping):
        target = {0: 0.10, 1: 0.80, 2: 0.05, 3: 0.05}
        return _make_sim(
            initial_datasets, train_edges, atc_mapping,
            events=[
                DriftEvent(
                    client_id=0,
                    start_round=10,
                    target_mixture=target,
                    kind="sudden",
                )
            ],
        ), target

    def test_mixture_before_start_round_unchanged(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, _ = self._make_sudden_sim(initial_datasets, train_edges, atc_mapping)
        pre_mix = sim.mixture_at(0, server_round=9)
        init_mix = _normalise(initial_datasets[0].class_mixture)
        for cls in init_mix:
            assert abs(pre_mix.get(cls, 0.0) - init_mix[cls]) < 1e-9

    def test_mixture_at_start_round_is_target(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, target = self._make_sudden_sim(initial_datasets, train_edges, atc_mapping)
        mix_at_10 = sim.mixture_at(0, server_round=10)
        norm_target = _normalise(target)
        for cls, p in norm_target.items():
            assert abs(mix_at_10.get(cls, 0.0) - p) < 1e-9

    def test_mixture_after_start_round_stays_at_target(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, target = self._make_sudden_sim(initial_datasets, train_edges, atc_mapping)
        norm_target = _normalise(target)
        for rnd in [10, 11, 25, 50]:
            mix = sim.mixture_at(0, server_round=rnd)
            for cls, p in norm_target.items():
                assert abs(mix.get(cls, 0.0) - p) < 1e-9

    def test_non_drifting_client_unaffected(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, _ = self._make_sudden_sim(initial_datasets, train_edges, atc_mapping)
        init_mix_c1 = _normalise(initial_datasets[1].class_mixture)
        for rnd in [1, 10, 50]:
            mix = sim.mixture_at(1, server_round=rnd)
            for cls, p in init_mix_c1.items():
                assert abs(mix.get(cls, 0.0) - p) < 1e-9

    def test_step_returns_edges_from_global_train_edges(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, _ = self._make_sudden_sim(initial_datasets, train_edges, atc_mapping)
        global_edges = _edge_set(train_edges)
        for rnd in [5, 10, 20]:
            for ds in sim.step(rnd):
                assert _edge_set(ds.edges).issubset(global_edges)

    def test_step_dataset_has_correct_mixture_after_drift(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, target = self._make_sudden_sim(initial_datasets, train_edges, atc_mapping)
        norm_target = _normalise(target)
        ds_post = sim.step(15)[0]  # client 0 after drift
        for cls, p in norm_target.items():
            assert abs(ds_post.class_mixture.get(cls, 0.0) - p) < 1e-9

    def test_step_dataset_has_initial_mixture_before_drift(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, _ = self._make_sudden_sim(initial_datasets, train_edges, atc_mapping)
        init_mix = _normalise(initial_datasets[0].class_mixture)
        ds_pre = sim.step(5)[0]  # client 0 before drift
        for cls, p in init_mix.items():
            assert abs(ds_pre.class_mixture.get(cls, 0.0) - p) < 1e-9


# ── Gradual drift ─────────────────────────────────────────────────────────────

class TestGradualDrift:
    def _make_gradual_sim(
        self, initial_datasets, train_edges, atc_mapping, transition_rounds=10
    ):
        target = {0: 0.10, 1: 0.80, 2: 0.05, 3: 0.05}
        sim = _make_sim(
            initial_datasets, train_edges, atc_mapping,
            events=[
                DriftEvent(
                    client_id=0,
                    start_round=20,
                    target_mixture=target,
                    kind="gradual",
                    transition_rounds=transition_rounds,
                )
            ],
        )
        return sim, target

    def test_mixture_before_start_is_initial(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, _ = self._make_gradual_sim(initial_datasets, train_edges, atc_mapping)
        init_mix = _normalise(initial_datasets[0].class_mixture)
        mix = sim.mixture_at(0, 19)
        for cls, p in init_mix.items():
            assert abs(mix.get(cls, 0.0) - p) < 1e-9

    def test_mixture_interpolates_during_transition(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, target = self._make_gradual_sim(
            initial_datasets, train_edges, atc_mapping, transition_rounds=10
        )
        init_mix = _normalise(initial_datasets[0].class_mixture)
        norm_target = _normalise(target)

        # At start_round (round 20), alpha = 0 → initial mixture
        mix_start = sim.mixture_at(0, 20)
        for cls, p in init_mix.items():
            assert abs(mix_start.get(cls, 0.0) - p) < 1e-9

        # At start + transition_rounds - 1 (round 29), alpha = 1 → target
        mix_end = sim.mixture_at(0, 29)
        for cls, p in norm_target.items():
            assert abs(mix_end.get(cls, 0.0) - p) < 1e-9

    def test_mixture_monotonically_moves_toward_target(
        self, initial_datasets, train_edges, atc_mapping
    ):
        """The dominant target class (class 1 at 80%) should have a
        monotonically increasing proportion during the transition."""
        sim, _ = self._make_gradual_sim(
            initial_datasets, train_edges, atc_mapping, transition_rounds=10
        )
        # Round 20 to 29 — during gradual transition
        proportions = [sim.mixture_at(0, r).get(1, 0.0) for r in range(20, 30)]
        for a, b in zip(proportions, proportions[1:]):
            assert b >= a - 1e-9  # non-decreasing

    def test_mixture_stays_at_target_after_transition(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, target = self._make_gradual_sim(
            initial_datasets, train_edges, atc_mapping, transition_rounds=5
        )
        norm_target = _normalise(target)
        # Round 24 = start(20) + transition(5) - 1 → should be at target
        for rnd in [24, 30, 50]:
            mix = sim.mixture_at(0, rnd)
            for cls, p in norm_target.items():
                assert abs(mix.get(cls, 0.0) - p) < 1e-9

    def test_result_always_sums_to_one(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim, _ = self._make_gradual_sim(
            initial_datasets, train_edges, atc_mapping, transition_rounds=10
        )
        for rnd in range(18, 35):
            mix = sim.mixture_at(0, rnd)
            assert abs(sum(mix.values()) - 1.0) < 1e-9


# ── Multiple drift events & multi-client drift ────────────────────────────────

class TestMultipleDriftEvents:
    def test_two_sudden_drifts_on_same_client(
        self, initial_datasets, train_edges, atc_mapping
    ):
        """Second sudden drift replaces the first — not the initial mixture."""
        target1 = {0: 0.1, 1: 0.9}
        target2 = {0: 0.6, 1: 0.3, 2: 0.1}
        sim = _make_sim(
            initial_datasets, train_edges, atc_mapping,
            events=[
                DriftEvent(0, start_round=10, target_mixture=target1, kind="sudden"),
                DriftEvent(0, start_round=20, target_mixture=target2, kind="sudden"),
            ],
        )
        norm_t1 = _normalise(target1)
        norm_t2 = _normalise(target2)

        mix_r10 = sim.mixture_at(0, 10)
        for cls, p in norm_t1.items():
            assert abs(mix_r10.get(cls, 0.0) - p) < 1e-9

        mix_r20 = sim.mixture_at(0, 20)
        for cls, p in norm_t2.items():
            assert abs(mix_r20.get(cls, 0.0) - p) < 1e-9

    def test_different_clients_drift_independently(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = _make_sim(
            initial_datasets, train_edges, atc_mapping,
            events=[
                DriftEvent(0, start_round=5,  target_mixture={0: 1.0}, kind="sudden"),
                DriftEvent(1, start_round=15, target_mixture={3: 1.0}, kind="sudden"),
                # client 2 has no event
            ],
        )
        # Round 6: client 0 has drifted, client 1 has not, client 2 is stable
        mix_c0_r6 = sim.mixture_at(0, 6)
        assert mix_c0_r6.get(0, 0.0) > 0.99  # class 0 dominant

        mix_c1_r6 = sim.mixture_at(1, 6)
        init_mix_c1 = _normalise(initial_datasets[1].class_mixture)
        for cls, p in init_mix_c1.items():
            assert abs(mix_c1_r6.get(cls, 0.0) - p) < 1e-9

        # Round 16: client 1 has now drifted
        mix_c1_r16 = sim.mixture_at(1, 16)
        assert mix_c1_r16.get(3, 0.0) > 0.99  # class 3 dominant

        # Client 2 is always stable
        init_mix_c2 = _normalise(initial_datasets[2].class_mixture)
        for rnd in [1, 10, 50]:
            mix_c2 = sim.mixture_at(2, rnd)
            for cls, p in init_mix_c2.items():
                assert abs(mix_c2.get(cls, 0.0) - p) < 1e-9

    def test_gradual_then_sudden_on_same_client(
        self, initial_datasets, train_edges, atc_mapping
    ):
        """A sudden drift following a gradual one overrides at the sudden
        start_round, regardless of whether the gradual transition completed."""
        target_gradual = {0: 0.05, 1: 0.90, 2: 0.05}
        target_sudden  = {3: 1.0}
        sim = _make_sim(
            initial_datasets, train_edges, atc_mapping,
            events=[
                DriftEvent(
                    0, start_round=10, target_mixture=target_gradual,
                    kind="gradual", transition_rounds=20,
                ),
                DriftEvent(0, start_round=15, target_mixture=target_sudden, kind="sudden"),
            ],
        )
        # At round 15, the sudden event takes over completely.
        mix_r15 = sim.mixture_at(0, 15)
        assert mix_r15.get(3, 0.0) > 0.99

        # And stays there.
        mix_r30 = sim.mixture_at(0, 30)
        assert mix_r30.get(3, 0.0) > 0.99


# ── Reproducibility & mutation safety ────────────────────────────────────────

class TestReproducibility:
    def test_step_reproducible_with_same_seed(
        self, initial_datasets, train_edges, atc_mapping
    ):
        events = [
            DriftEvent(0, start_round=5, target_mixture={1: 1.0}, kind="sudden")
        ]
        sim_a = ConceptDriftSimulator(
            initial_datasets, train_edges, atc_mapping, events=events,
            observations_per_client=100, seed=42,
        )
        sim_b = ConceptDriftSimulator(
            initial_datasets, train_edges, atc_mapping, events=events,
            observations_per_client=100, seed=42,
        )
        for rnd in [1, 5, 10]:
            for ds_a, ds_b in zip(sim_a.step(rnd), sim_b.step(rnd)):
                assert torch.equal(ds_a.edges, ds_b.edges)

    def test_different_seeds_produce_different_samples(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim_a = ConceptDriftSimulator(
            initial_datasets, train_edges, atc_mapping, seed=1,
        )
        sim_b = ConceptDriftSimulator(
            initial_datasets, train_edges, atc_mapping, seed=2,
        )
        ds_a = sim_a.step(1)
        ds_b = sim_b.step(1)
        assert not torch.equal(ds_a[0].edges, ds_b[0].edges)

    def test_mixture_at_does_not_mutate_simulator_state(
        self, initial_datasets, train_edges, atc_mapping
    ):
        """mixture_at() is a pure query — it must not update _current_mixtures
        or _gradual_start_mixtures. Calling step() after mixture_at() must
        give the same result as if mixture_at() was never called."""
        events = [
            DriftEvent(0, start_round=10, target_mixture={2: 1.0}, kind="sudden")
        ]
        sim_ref = ConceptDriftSimulator(
            initial_datasets, train_edges, atc_mapping, events=events,
            observations_per_client=100, seed=99,
        )
        sim_query = ConceptDriftSimulator(
            initial_datasets, train_edges, atc_mapping, events=events,
            observations_per_client=100, seed=99,
        )

        # Call mixture_at on a FUTURE round on sim_query — should not change state.
        _ = sim_query.mixture_at(0, 50)
        _ = sim_query.mixture_at(1, 50)

        # Now both simulators step through the same sequence.
        for rnd in [1, 5, 10, 15]:
            ref_ds  = sim_ref.step(rnd)
            query_ds = sim_query.step(rnd)
            for ds_r, ds_q in zip(ref_ds, query_ds):
                # Mixtures must be identical.
                for cls in set(ds_r.class_mixture) | set(ds_q.class_mixture):
                    assert abs(
                        ds_r.class_mixture.get(cls, 0.0)
                        - ds_q.class_mixture.get(cls, 0.0)
                    ) < 1e-9

    def test_current_mixtures_property_returns_snapshot(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = _make_sim(initial_datasets, train_edges, atc_mapping)
        snap1 = sim.current_mixtures
        snap2 = sim.current_mixtures
        # Should be independent copies, not the same object.
        assert snap1 is not snap2
        assert snap1[0] is not snap2[0]


# ── Edge-correctness after drift ──────────────────────────────────────────────

class TestEdgesAfterDrift:
    def test_all_edges_in_global_train_set_after_sudden_drift(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = _make_sim(
            initial_datasets, train_edges, atc_mapping,
            events=[
                DriftEvent(0, start_round=5, target_mixture={0: 1.0}, kind="sudden")
            ],
        )
        global_edges = _edge_set(train_edges)
        for rnd in [1, 5, 10, 30]:
            for ds in sim.step(rnd):
                bad = _edge_set(ds.edges) - global_edges
                assert not bad, f"Round {rnd}: invented edges {bad}"

    def test_observation_count_matches_n_obs(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = ConceptDriftSimulator(
            initial_datasets, train_edges, atc_mapping,
            events=[
                DriftEvent(0, start_round=3, target_mixture={1: 1.0}, kind="sudden")
            ],
            observations_per_client=75,
            seed=0,
        )
        for rnd in [1, 3, 10]:
            for ds in sim.step(rnd):
                assert len(ds) == 75

    def test_class_agnostic_fallback_works_without_atc_mapping(
        self, initial_datasets, train_edges
    ):
        """Drift simulation must not crash when atc_mapping is None."""
        datasets_no_atc = generate_client_observations(
            train_edges, num_clients=NUM_CLIENTS,
            atc_mapping=None, observations_per_client=50, seed=0,
        )
        sim = ConceptDriftSimulator(
            datasets_no_atc, train_edges, atc_mapping=None,
            events=[
                DriftEvent(0, start_round=5, target_mixture={0: 1.0}, kind="sudden")
            ],
            observations_per_client=50,
            seed=0,
        )
        global_edges = _edge_set(train_edges)
        for rnd in [1, 5, 10]:
            for ds in sim.step(rnd):
                assert _edge_set(ds.edges).issubset(global_edges)


# ── Simulator construction validation ────────────────────────────────────────

class TestSimulatorValidation:
    def test_empty_initial_datasets_raises(self, train_edges, atc_mapping):
        with pytest.raises(ValueError, match="empty"):
            ConceptDriftSimulator([], train_edges, atc_mapping)

    def test_mixture_at_out_of_range_client_raises(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = _make_sim(initial_datasets, train_edges, atc_mapping)
        with pytest.raises(ValueError, match="out of range"):
            sim.mixture_at(NUM_CLIENTS, server_round=1)

    def test_mixture_at_negative_client_raises(
        self, initial_datasets, train_edges, atc_mapping
    ):
        sim = _make_sim(initial_datasets, train_edges, atc_mapping)
        with pytest.raises(ValueError):
            sim.mixture_at(-1, server_round=1)
