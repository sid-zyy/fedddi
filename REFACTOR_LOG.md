# Refactor Log — FedDrift transition (graph partitioning → shared global graph)

This log tracks the step-by-step transition away from the graph-partitioning
design (see `STRATEGY_ANALYSIS.md` / `NEXT_STEPS.md`, now marked historical)
towards the FedDrift research direction: every client trains on the same
shared global DDI graph, and heterogeneity/privacy comes from each client's
own private local clinical observations rather than graph ownership.

Each step below was verified before moving to the next one via:
  1. `python -m py_compile <changed files>` — syntax check
  2. `python -c "import main_fl"` — import-level sanity check
  3. `python -m pytest tests/ -q` — full unit test suite
  4. An end-to-end synthetic-data smoke test (`smoke_test_fl.py`, dev-only,
     not part of the deliverable) that runs `main_fl.main()`'s REAL Flower
     + Ray FL simulation against a tiny synthetic graph, since this sandbox
     has no network access to download the real ogbl-ddi dataset from
     snap.stanford.edu.

---

## Step 1 — Remove obsolete graph partitioning
**Status: done.**

- Deleted `src/data/partitioning.py` and `tests/test_partitioning.py` in
  full (ATC partitioning, Dirichlet node partitioning, full-graph
  partitioning, induced-subgraph construction — no METIS or halo-node code
  existed in this codebase).
- `main_fl.py`: removed `partition.strategy` dispatch; every client now
  gets `graph = global_edge_index` directly. Removed the `in_client`
  node-ownership mask that used to filter training edges per client.
- `fl_client.py`: removed the `local_Hits@20` / node-ownership evaluation
  path (meaningless once every client sees every node by construction).
- `conf/config.yaml`: removed the `partition:` block.
- Removed/updated stale documentation describing the deleted system
  (`README.md`; `STRATEGY_ANALYSIS.md` and `NEXT_STEPS.md` marked as
  historical records of the pre-refactor debugging process).
- Removed the now-unused `skewed_atc_mapping` test fixture.

**Verification:** 46/46 tests pass (74 → 46 after deleting 28
partitioning-only tests). End-to-end synthetic FL simulation runs cleanly.

## Step 2 — Simplify client creation
**Status: done.**

- Extracted the inline per-client model/optimizer construction (previously
  a for-loop inside `main_fl.main()`) into one `build_client_trainer()`
  factory function in `src/training/trainer.py`.
- This makes the federated-client contract explicit: every client gets the
  **identical** shared `graph` and `node_features`, but its **own**
  `GraphSAGEEncoder`, `NeuralLinkPredictor`, optimizer (scoped to only its
  own parameters), and own local training observations.
- Added `tests/test_client_creation.py` (7 tests) asserting this contract
  directly, including that one client's local training step never mutates
  another client's weights.

**Verification:** 53/53 tests pass (46 + 7 new). End-to-end synthetic FL
simulation runs cleanly.

## Step 3 — Keep GraphSAGE unchanged
**Status: done (verification-only — no code changed).**

- Confirmed via `git diff` against the pre-refactor baseline commit that
  `src/models/encoder.py` and `src/models/predictor.py` are **byte-for-byte
  identical** to before Step 1 — no architecture, message-passing,
  embedding, or decoder changes anywhere in this refactor.
- The existing `tests/test_encoder.py` / `tests/test_predictor.py` (22
  tests total) already pin output shapes, SAGEConv layer count and
  parameter scaling, embedding trainability, gradient flow, and
  reproducibility under fixed seeds — all 22 still pass unmodified.

**Verification:** `git diff <baseline>..HEAD -- src/models/encoder.py
src/models/predictor.py` produces no output. 53/53 tests pass.

## Step 4 — Implement heterogeneous observations
**Status: done.**

- Added `src/data/local_observations.py`:
  - `LocalObservationDataset` — a small dataclass representing one client's
    private local clinical observations (a multiset of training edges +
    the ATC-class mixture it was sampled from). NOT a graph partition:
    clients may share edges, and the dataclass has no notion of node
    ownership at all.
  - `generate_client_observations()` — draws each client a personalized
    mixture over ATC Level-1 therapeutic classes from a symmetric
    Dirichlet(alpha) distribution, then samples that client's local
    observations (with replacement) from the global training edges
    according to its mixture. Falls back gracefully to class-agnostic
    sampling if no ATC mapping is available.
  - `sample_observations()` — the shared low-level primitive (sample N
    edges given an explicit target mixture), reused as-is by the
    concept-drift simulator in Step 5.
- `main_fl.py`: every client now gets its own `LocalObservationDataset`
  instead of the identical shared `train_edges`. Reintroduced a
  best-effort (non-fatal) ATC mapping load — used for observation
  sampling, never for partitioning. The message-passing `graph` passed to
  every client is completely unaffected by this change.
- `conf/config.yaml`: added a `local_observations:` block
  (`dirichlet_alpha`, `observations_per_client`, `seed`).
- Added `tests/test_local_observations.py` (14 tests): every client's
  observations are a verified subset of the real global edges, different
  clients get measurably different class mixtures under a skewed alpha,
  results are reproducible under a fixed seed, and the class-agnostic
  fallback works without an ATC mapping.

**Verification:** 67/67 tests pass (53 + 14 new). End-to-end synthetic FL
simulation runs cleanly, now exercising the real ATC-aware path (verified
via log output showing two clients converging on different dominant
therapeutic classes, e.g. client 0 → 55% class 2, client 1 → 77% class 1).

## Step 5 — Introduce concept drift
**Status: done.**

- Added `src/data/drift.py`:
  - `DriftEvent` — a dataclass specifying one drift event: which client
    drifts, at which round, to what target mixture, and with what schedule
    (`"sudden"` or `"gradual"`). Validated at construction.
  - `ConceptDriftSimulator` — owns the per-client current mixture and
    re-samples each client's `LocalObservationDataset` every round via
    `step(server_round)`. Calls `sample_observations()` from Step 4 —
    no new sampling logic. The graph is NEVER touched.
  - `_normalise()` / `_interpolate()` — pure helpers for mixture
    arithmetic, independently testable.
  - **Sudden drift**: the target mixture replaces the initial one at
    exactly `start_round`. Subsequent rounds hold the target.
  - **Gradual drift**: the mixture linearly interpolates from the start
    to the target over `transition_rounds` rounds, then holds the target.
  - Multiple events on the same client compose correctly (chained events
    use the mixture at the moment the previous event fired as the new
    "start", not the original initial mixture).
  - Multiple clients can drift independently and on different schedules.
  - `mixture_at(client_id, round)` — pure query: returns the effective
    mixture at any round without mutating simulator state.
  - Reproducible under a fixed seed; per-round RNG derived deterministically.
- `main_fl.py`: drift simulator is constructed from a `drift.events`
  config list (empty by default = stable FedAvg baseline). A
  `round_tracker` dict bridges Flower's `client_fn` (which has no
  `round` argument) to the simulator's `step()` call. A
  `_drift_aware_fit_agg` wrapper advances the tracker before each fit
  round so every client gets the correct-round observations.
- `conf/config.yaml`: added a `drift:` block with `seed` and `events: []`
  (disabled by default). The block includes commented-out examples showing
  both sudden and gradual event syntax.
- `src/data/__init__.py`: exports `ConceptDriftSimulator` and `DriftEvent`.
- Added `tests/test_drift.py` (42 tests): normalise/interpolate unit
  tests, DriftEvent validation, no-drift baseline, sudden drift
  (pre/post/held), gradual drift (interpolation, monotonicity, target
  lock, sums-to-one), multi-event chaining, multi-client independence,
  edge-correctness invariants, reproducibility, mutation safety of
  `mixture_at()`, and constructor validation.

**Verification:** 101/101 tests pass (59 pre-existing + 42 new).
Import-level sanity check clean. Graph is never mutated across all test
scenarios.
