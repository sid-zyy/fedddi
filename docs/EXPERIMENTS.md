# Experiment protocol

`scripts/journal_protocol.py` defines the protocol; `scripts/run_experiments.py` provides its command-line interface.

| Setting | Value |
| --- | --- |
| Clients | 10 |
| Local observations per client | 200,000 |
| Rounds / local epochs | 100 / 5 |
| Seeds | 0–4 |
| Poisoned clients | 0, 1 |
| Drifted clients | 2, 3 |
| Onset rounds | 40 and 50 |
| Evaluation window | Rounds 81–100 |

Poisoned and drifted client groups are disjoint. Conditions are clean, poison-only, drift-only and joint. Tier 1 compares FedAvg and CAGA on clean and joint conditions; tier 2 adds norm clipping; tier 3 compares all three methods on poison-only and drift-only conditions. The complete protocol contains 60 runs. Optional tier 4 runs FLTrust on all four conditions (20 further runs); it is not selected by default and must be requested with `--tiers 4`.

The protocol's reference window is rounds 30–39 inclusive, strictly before the round-40 onset. If you change `attack_start`, move both windows with it and use the same choice for every method and seed.

## Running and resuming

The runner lists planned jobs by default. Add `--execute` to run them. Each experiment writes a log and per-round CSV under its own results directory. Only runs whose CSV reaches the final round count as complete. Reusing a results directory skips completed runs; incomplete runs restart and overwrite their log. Use a new directory when changing the protocol or hyperparameters.

The CLI uses the standard 100-round protocol. For installation checks, use the two-round `main_fl.py` command in the README. The full protocol expects CUDA. Clients run sequentially on one GPU; measure pilot runtime before scheduling all 60 runs.

Performance flags remain opt-in. Run `tests/test_perf_equivalence.py` and `python -m scripts.bench_perf` before adding `--perf`. The optimized and standard paths should be checked on the intended hardware.

## Reporting

`--summarize` writes window metrics, seed summaries, paired bootstrap comparisons and routing summaries. Compare methods on matching conditions and seeds. Report global Hits@20 alongside fabricated-edge probabilities and drift-region Hits@20; a global metric can conceal targeted harm.

Keep raw logs, per-round CSVs and the exact configuration with each published result. These generated outputs are excluded from the source repository. No previous experimental numbers are presented as validated results in this cleaned version.

## Additional utilities

- `scripts/verify_poisoning_application.py`: inspect the application of configured label attacks.
- `scripts/verify_data_drift_application.py`: inspect feature perturbations.
- `probe_vram.py`: measure training memory for the selected configuration.
- `scripts/prepare_features.py`: prepare optional chemical features for the centralized legacy-feature mode; not required for CAGA's fixed structural features.
