# Clean-Anchor Gated Aggregation

CAGA measures candidate updates on a fixed clean server probe within each round. Its objective is a pairwise ranking loss over positive and negative edges. The NumPy implementation receives this per-sample loss through a callback; the PyTorch adapter evaluates it using the shared graph and clean server features.

## Aggregation

1. Form client deltas from the previous global parameters.
2. Start from ordinary sample-weighted averaging.
3. Try shrinking or removing each client's contribution. Keep a change only when it improves aggregate probe loss beyond the paired standard-error tolerance and absolute floor.
4. Repeat this search for the feature projection, convolution and MLP blocks when block gating is enabled.
5. Select a final aggregate step size using the same probe: the largest step not significantly worse than the best. `caga_max_rel_increase` (default 0.1) is a hard cap: a step whose mean probe loss exceeds the best by more than that fraction is never taken. Without it, the heavy-tailed losses of a diverging model inflate the standard error enough to pass the noise test. `caga_tol_rel` does not loosen this check.

Weights are renormalized separately for each block over clients whose block contribution remains positive. Comparing aggregate candidates matters: an honest non-IID client's update can be poor in isolation and useful in combination.

`caga_tol_z` controls the standard-error tolerance, `caga_tol_abs` provides an absolute floor and `caga_tol_rel` adds a materiality threshold relative to the round's initial aggregate loss change. `caga_probe_size` and `caga_max_sweeps` control scoring cost. See `CAGAHyperparams` in `src/training/caga.py` for the complete search configuration.

## Server probe

The server reserves a fraction of the training edges before clients sample their observations. Probe positives come from this pool; negatives are sampled against the root edges. Root edges remain part of the public message-passing graph. Negatives are sampled against the whole public training graph, so no known training interaction is used as a probe negative.

Each candidate in a round is scored on the same probe. The round number seeds both positive and negative sampling without disturbing the global random state, so a run's probes are reproducible. Across candidates, only parameters change: the probe uses the server's clean features.

## Limitations

- Improving probe loss does not guarantee improved test Hits@20 or protection of every targeted edge.
- Attacks below the per-round tolerance can accumulate.
- A stale clean probe can reject useful adaptation to a genuine distribution change.
- Gating an entire client removes its useful signal as well as its harmful signal.
- Routing labels describe the selected gating action; they are not validated poison-versus-drift classifications.
- Candidate scoring adds model forward passes. Runtime and memory must be measured on the intended hardware.

## FLTrust baseline

`defense.mode=fltrust` implements FLTrust (Cao et al., NDSS 2021) for comparison. It reserves the same root edges as CAGA. Each round the server trains a copy of the global model on those edges with the clients' local-training procedure. Each client receives trust `ReLU(cos(g_i, g_0))`, where `g_0` is the server update. Each client update is rescaled to `‖g_0‖`, and the global step is their trust-weighted mean. Sample counts are not used. If no client has positive trust, the global model is unchanged for that round.

Differences from CAGA: FLTrust compares update directions over the whole parameter vector rather than probe losses, has no block gating, and adds one server training pass per round instead of candidate forward passes. Its root set here is about 12% of training edges, far larger than the 100-example root set in the original paper. Because client updates are rescaled to the server update's norm, the server trains each round on a fresh sample of root edges the size of a client's observations (`defense.fltrust_root_obs`), so it takes as many optimizer steps as a client. Otherwise a root set larger than a client's sample inflates every step, and a smaller one shrinks it. Report `root_frac` with results.

The synthetic quadratic simulation in `scripts/caga_sim.py` checks selection behaviour. Real-data conclusions require the paired experiment protocol and retained raw run outputs.
