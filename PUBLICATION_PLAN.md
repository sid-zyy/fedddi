# Publication plan

The plan for turning FedDDI-CAGA into a journal paper. Tick items off as they are done, and record decisions here so co-authors share one picture.

Last updated: 2026-10-12.

## Thesis

**One probe-gated aggregation rule protects a federated model from poisoning and feature drift at the same time, without knowing which threat is present.**

Decided 2026-10-12 (Option A). An earlier version claimed CAGA routes the two threats to different blocks: poisoned clients rejected, drifted clients gated only on the feature projection. The CPU pilot did not support that mechanism. Under severe drift, drifted clients were scaled down or rejected and never feature-gated, because shifted inputs also change what the convolutions and MLP learn. Block gating is now an ablation, not the contribution.

What CAGA does in one rule:

| Threat | What is corrupted | What CAGA did in the pilot |
| --- | --- | --- |
| Poisoning | Labels (fabricated or suppressed interactions), sometimes boosted updates | Poisoned clients accepted in 3% of post-onset rounds; the rest scaled or rejected |
| Severe feature drift | Node features; labels are still correct | Drifted clients scaled down (34–53%) or rejected (16–32%); test Hits@20 recovered to clean-FedAvg level |

**Contributions to establish, each with its own evidence:**

1. **Joint robustness.** In the joint condition CAGA beats FedAvg, norm clipping, FLTrust and the root-client control, with paired confidence intervals that exclude zero.
2. **No threat model needed.** The same settings work under poison-only, drift-only and joint conditions, with no per-threat tuning.
3. **Combination search.** CAGA searches over combinations of clients rather than scoring each client in isolation. A non-IID honest client can look harmful alone and still help the combined update. Ablation: per-client scoring (Fang et al.'s loss-based rejection).
4. **Bounded damage per round.** The final step-size check with its hard cap limits what one round can do.

**Must be reported, not hidden:** CAGA also beats FedAvg with no attack (+0.036 in the pilot), and adding the root data to FedAvg does not explain this. Separate this non-IID aggregation gain from the defense gain by comparing each method's attacked score with its own clean score.

**Ablations:** block gating on vs. off (`caga_noblock`); final check on vs. off; per-client vs. combination search.

## What "handling drift" means

Meaning used (**drift as corruption**): protect the shared model, evaluated with clean features, from clients whose features have drifted. CAGA's clean probe is built for this, and it does so by down-weighting drifted clients, which also discards their correct label signal.

Not claimed (**drift as a genuine change in the data**): serving drifted clients well on their shifted features. A fixed clean probe works against this. A possible extension is personalization, where down-weighted clients keep local copies of some layers (in the style of FedBN or FedPer); it is out of scope unless the main results are strong.

Metrics:

- **Primary:** clean test Hits@20.
- **Attack-specific:** mean probability on fabricated edges (lower is better) and on suppressed true edges (higher is better), reported separately.
- **Reported:** drift-region Hits@20, which drifted clients measure on their own shifted features. It collapses under severe drift for every method; explain this in the paper.
- **Routing:** how often poisoned, drifted and honest clients are accepted, scaled or rejected, including false gating of honest clients.

## Where we are

### Evidence so far (pilot runs, 2026-10-06 and 2026-10-07)

Five 100-round joint-condition runs. They used an older protocol: onset at round 60, 400k observations per client, strong poisoning, `caga_tol_rel=0.1`, lr 0.005. Zip 1 used batch size 16384 and older code; zip 2 used batch size 8192. Treat them as diagnostics, not results.

- **Poison detection works.** After onset, both CAGA runs scaled down or rejected poisoned client 0 in every round from 60 to 69. In zip 1 they also caught poisoned client 1 in 90% of rounds from 70 on. Honest clients were fully accepted in at least 90% of rounds.
- **Drift routing did not happen.** Drifted clients 2 and 3 were fully accepted in at least 90% of post-onset rounds. Their feature projection was gated in 1 of 26 client-rounds. The core claim is so far untested or failing.
- **Only one run held up.** In zip 1, CAGA kept test Hits@20 at 0.10 in rounds 81–100, while both FedAvg seeds collapsed to 0. That is one seed, with Hits@20 swinging between 0.01 and 0.21 from round to round.
- **Training is unstable without any attack.** Norm clipping diverged at round 40, before onset, and zip 2's CAGA had loss spikes before onset. Collapsed runs show a training loss of exactly 2·ln 2 = 1.386: the model predicts 0.5 for every pair and cannot recover.
- **Too much false gating before the attack.** Before onset, some honest clients were not fully accepted in 30–70% of rounds.

### CPU pilot (2026-10-12)

Setup: `scripts/cpu_pilot.py` with the repository's client, attack and aggregation code, scaled down (hidden size 128, 20k observations per client, 40 rounds, onset at round 12, moderate poisoning, 2 seeds). Window: mean test Hits@20 over rounds 31–40. Treat this as an early signal only; with 2 seeds there are no meaningful confidence intervals.

| Condition | FedAvg | Norm clip | FLTrust | CAGA |
| --- | --- | --- | --- | --- |
| Clean | 0.103 | — | — | 0.139 |
| Drift only | 0.097 | — | — | 0.115 |
| Joint | 0.076 | 0.096 | 0.116 | 0.127 |

- **Poisoning: promising.** In the joint condition CAGA beat FedAvg on both seeds (+0.060, +0.042). It accepted poisoned clients in 3% of post-onset rounds, scaling or rejecting the rest, while accepting honest clients in 82%. Its lead over FLTrust was +0.021 on seed 0 and +0.001 on seed 1, which is not yet distinguishable.
- **Part of CAGA's lead is not defense.** CAGA also beat FedAvg under clean conditions (+0.036 on average). It uses the clean root edges to select updates, which FedAvg cannot. Needed baseline: FedAvg with the server training on the root edges as an extra client.
- **Drift: the thesis is not supported yet.** Drift as configured barely hurt FedAvg (0.097 vs. 0.103 clean, within seed noise), so there was little to handle. Drifted clients were routed to `phi_gated` in only 5–18% of post-onset rounds. They were more often fully rejected, in 5–21% of rounds.
- **False gating early in training.** Before onset, honest clients were fully accepted in only 26–48% of rounds; this rose to 65–84% later.
- **The poison metric mixes directions.** `poison_attack_mean_prob` averages fabricated edges (lower is better) with suppressed true edges (higher is better). Report them separately.
- **Cost.** CAGA rounds took 2–3× as long as FedAvg rounds on CPU.

### CPU pilot, step 1 (2026-10-12)

Same setup, 2 seeds. Drift levels: base (4 features, 25% of nodes), wide (16 features, +2.0, noise 0.5, 50% of nodes), severe (all 50 features, +2.0, noise 1.0, 75% of nodes).

| Condition | FedAvg | FedAvg + root client | CAGA |
| --- | --- | --- | --- |
| Clean | 0.103 | 0.109 | 0.139 |
| Drift, base | 0.097 | — | 0.115 |
| Drift, wide | 0.104 | — | 0.110 |
| Drift, severe | **0.073** | — | **0.107** |
| Joint | 0.076 | 0.075 | 0.127 |

- **Only severe drift hurts.** FedAvg lost 0.031 and 0.030 on the two seeds; base and wide drift had no clear effect. Severe drift is the level to use for drift claims.
- **CAGA mitigates severe drift,** beating FedAvg by +0.035 and +0.032, back to roughly clean-FedAvg level.
- **But not through block routing.** Under severe drift, drifted clients were never feature-gated (0% `phi_gated`). They were scaled down (34–53% of post-onset rounds) or rejected (16–32%). The premise that drift harm stays in the feature projection did not hold: drifted inputs change what the convolutions and MLP learn too.
- **Root data alone does not explain CAGA's lead.** Adding the server's root update as an extra FedAvg client gained little (+0.006 clean, −0.001 joint). CAGA's advantage comes from probe-guided selection, which also helps under clean conditions (+0.036). That clean-condition gain must be reported, and separated from the defense effect in the paper.

**Decision (2026-10-12): Option A.** The thesis is now the single probe-gated rule (see Thesis). Severe drift is the protocol default (`drift_strength` in `scripts/journal_protocol.py`). Running: CAGA with and without block gating (`caga_noblock`, protocol tier 5) under severe drift and the joint condition, plus FedAvg, norm clipping and FLTrust in the joint condition with severe drift.

### CPU pilot, Option A check (2026-10-12)

Joint condition with severe drift, 2 seeds, test Hits@20 over rounds 31–40:

| Method | Seed 0 | Seed 1 | Mean |
| --- | --- | --- | --- |
| FedAvg | 0.076 | 0.079 | 0.077 |
| Norm clip | 0.077 | 0.087 | 0.082 |
| FLTrust | 0.112 | 0.121 | 0.116 |
| CAGA without block gating | 0.144 | 0.145 | 0.145 |
| **CAGA** | **0.173** | **0.139** | **0.156** |

- **Phase 2 gate passed at pilot level:** CAGA beat FLTrust on both seeds (+0.061, +0.019) and FedAvg by +0.097 and +0.060. Two seeds are not enough for a confidence interval.
- **Detection is clean in the joint condition:** after onset CAGA fully accepted poisoned clients in 3% of rounds and drifted clients in 3–13%, and honest clients in 88–94%. False gating before onset remains high (honest clients fully accepted in about 41% of rounds).
- **Targeted metrics:** fabricated-edge probability 0.186 for CAGA vs. 0.226 for FedAvg and 0.183 for FLTrust (lower is better); suppressed-edge probability 0.845 vs. 0.821 and 0.858 (higher is better). CAGA and FLTrust are similar here.
- **Block gating is inconclusive.** Joint: +0.029 on seed 0, −0.006 on seed 1. Severe drift only: no difference (0.089 vs. 0.087, 0.124 vs. 0.125). Without block gating, rounds were about 40% faster (85 s vs. 140 s). Decide with more seeds during tuning; if there is still no difference, default to whole-client gating.
- **Caution:** CAGA in the joint condition (0.156) scored above CAGA under clean conditions (0.139). With 6 honest clients left, discarding the 4 attacked clients cost nothing here, so the argument that CAGA keeps useful signal from attacked clients is not supported by this setup. The gap is also within the round-to-round noise of Hits@20.
- **Drift-region Hits@20 collapses for every method** under severe drift (≤0.04), as expected when drifted clients evaluate on their own shifted features.

### Fixed on 2026-10-11

- [x] **Metric dilution.** Drift-region Hits@20 and attack probability were divided by all 10 clients' example counts but reported by only 2 clients, so they were under-reported by 5×. The "0.1000" attack probability after collapse was really 0.5. Each metric is now averaged only over the clients that report it (`make_metrics_aggregation_fn`, `tests/test_metrics_aggregation.py`). Older CSVs need drift and poison metrics multiplied by `num_clients / reporting clients` before reuse.
- [x] **Divergent step accepted.** In zip 2, round 73, CAGA's final check accepted a step that took the probe loss from 0.22 to 87.07, and the model never recovered. Two causes:
  - The heavy-tailed losses of a diverging model inflate the standard error, so the noise test passed.
  - `tol_rel` also loosened the final check, by an amount that grows with the damage done.

  Fix: added a hard cap, `caga_max_rel_increase` (default 0.1), and `tol_rel` no longer affects the final check. Regression tests in `tests/test_caga.py` reproduce the failure.
- [x] **Probe negatives.** They could include known interactions and depended on the global random state. They now exclude the whole public training graph and are seeded by the round number.
- [x] **Reference window.** It included the onset round; it is now rounds 30–39.
- [x] **FLTrust baseline added** (`defense.mode=fltrust`, protocol tier 4).

## Phase 1: Trustworthy training (1–2 weeks)

- [ ] Put the code under git and tag the commit used for every reported run. The current folder is not a repository.
- [ ] Pin dependency versions in a lockfile; the current requirements are ranges only.
- [ ] Choose a stable learning rate and batch size: one clean FedAvg run for 100 rounds with no divergence. Freeze them for the rest of the paper.
- [ ] Run the centralized baseline (`train.py baseline.use_fixed_features=true`) to measure the ceiling. Explain the gap to published `ogbl-ddi` results, splitting the cost of fixed features from the cost of federation.
- [ ] Decide how a round's metric is reported: window mean, last-K mean, or best-validation checkpoint. Use the same choice everywhere.
- [ ] Run the GPU test suite (`python -m pytest tests/ -m "not slow and not integration"`) on the training machine, since the probe and FLTrust changes have only been checked without torch.

## Phase 2: Settle the design and tune (CPU, before GPU time)

- [x] **Drift hurts.** Only severe drift (all 50 features, 75% of nodes) measurably hurt FedAvg; it is now the protocol default.
- [x] **Root-data control.** FedAvg with the server's root update as an extra client gained little, so CAGA's lead is not just extra data.
- [x] **Thesis decided:** Option A, one probe-gated rule for both threats (see Thesis).
- [x] **Block-gating ablation** (`caga_noblock`): inconclusive at 2 seeds; revisit with a third seed during tuning.
- [x] **Joint condition with severe drift:** CAGA 0.156, FLTrust 0.116, norm clip 0.082, FedAvg 0.077 (2 seeds). Gate passed at pilot level.
- [x] **Shared training settings** (CPU scale, 2026-10-13): grid of lr {0.001, 0.003, 0.01} × local epochs {1, 2, 4} on clean FedAvg, tuning seeds 10–11, validation Hits@20 over rounds 31–40. Chosen: **lr 0.003, 2 local epochs** (0.126). It tied lr 0.001 with 1 epoch (0.126), but was still improving (+0.024 over the previous 10 rounds, against +0.008), which matters for the 100-round protocol. It also reached a lower training loss and matches every earlier pilot run. lr 0.01 was unstable. **More local epochs lowered training loss but hurt validation:** 4 epochs was worst at lr 0.003 (0.079). The GPU protocol's 5 local epochs take a similar number of optimizer steps per round, so check 2 vs. 5 local epochs in the GPU validation runs.
- [ ] **CAGA tuning, interim (2026-10-13).** Joint condition with severe drift, seeds 10–11, validation Hits@20 over rounds 31–40:

  | Setting | Val (seed 10 / 11) | Mean | Honest fully accepted, pre / post onset | Poisoned / drifted accepted post onset |
  | --- | --- | --- | --- | --- |
  | `tol_z` 1 | 0.129 / 0.174 | 0.151 | 9% / 55% | 0% / 1% |
  | Warm-up 5 | 0.140 / 0.134 | 0.137 | 64% / 79% | 1% / 5% |
  | Warm-up 10 | 0.132 / 0.137 | 0.135 | 94% / 87% | 4% / 7% |
  | `tol_rel` 0.1 | 0.151 / 0.108 | 0.130 | 82% / 99% | 8% / 37% |
  | `tol_z` 3 | 0.114 / 0.131 | 0.123 | 55% / 86% | 5% / 4% |
  | No block gating | 0.119 / 0.101 | 0.110 | 64% / 83% | 5% / 13% |
  | Defaults | 0.124 / 0.081 | 0.102 | 27% / 71% | 3% / 3% |

  FedAvg on seed 10 scored 0.120, and norm clipping at multiplier 1.0 scored 0.144; the seed-11 baselines are still running. **On these seeds the default CAGA did not beat FedAvg**, unlike on seeds 0–1. Seed-to-seed gaps reach 0.045, so differences under about 0.02 are noise.

  Leading candidate: **warm-up 5**. It improved both seeds, is the most consistent, and fixes the documented failure (honest clients fully accepted before onset went from 27% to 64%) while still catching attacked clients. `tol_z` 1 had the highest mean but gates honest clients heavily (55% fully accepted after onset), had the largest seed gap, and spiked training loss to 1.67. `tol_rel` 0.1 let drifted clients through in 37% of rounds. Warm-up assumes attacks do not start in the first rounds; test an attack from round 1 before the paper.

  Block gating across 4 seeds (CAGA minus no-block): +0.029, −0.006, +0.005, −0.020, mean about 0, and no-block rounds are about 35% faster. Confirmation running: warm-up 5 without block gating on seeds 10–11.

- [ ] **CAGA tuning** on separate tuning seeds, validation Hits@20 only: `tol_z`, `tol_rel`, probe size, and possibly a no-gating warm-up. Targets: false gating of honest clients early in training, and the clean-vs-attacked gap.
- [ ] **Baseline tuning** with a comparable budget: norm clip multiplier, FLTrust root sample size.
- [ ] **Freeze** settings and write the pass criteria here before the GPU matrix.

**Gate:** do not start Phase 3 until CAGA beats FLTrust in the pilot's joint condition with severe drift, and settings are frozen.

## Phase 3: Main experiments

**Design:** a 2×2 factorial with the conditions clean, poison-only, drift-only and joint, at least 5 seeds, and paired comparisons with bootstrap confidence intervals. The joint condition is where the thesis is tested.

**Methods:**

| Method | Status | Why it is needed |
| --- | --- | --- |
| FedAvg, FedProx | done | Undefended references |
| Median-norm clipping | done | Simple magnitude defense |
| FLTrust | done | Uses clean server data, but only weights whole clients |
| Fang et al. loss-based rejection (LFR) | to do | **Closest prior work**; rejects whole clients by validation loss |
| Krum or Multi-Krum, coordinate median, trimmed mean | to do | Standard robust-aggregation baselines |
| FLAME | optional | Recent clustering-based defense |
| CAGA without block gating (`caga_noblock`) | running in pilot | Shows whether per-block gating adds anything beyond whole-client gating |
| CAGA scoring clients individually | to do | Tests the claim about searching over combinations |

**Attacks.** The current attacks are label poisoning plus update scaling. Add:

- [ ] Model-poisoning attacks: "A Little Is Enough"; min-max and min-sum (Shejwalkar & Houmansadr, NDSS 2021).
- [ ] An **adaptive attacker** that knows CAGA and stays just under its tolerance each round. Our own docs say such attacks can accumulate, so measure how much.
- [ ] Drift of different strengths and shapes (feature subsets, node fractions, gradual vs. sudden), to map where CAGA stops helping.

**Metrics:**

- Clean test Hits@20, the primary metric.
- Drift-region and non-drift-region Hits@20.
- Attack success: mean probability on fabricated edges, and Hits@20 on suppressed edges.
- **Routing statistics** over rounds: how often poisoned, drifted and honest clients are accepted, scaled or rejected. Report detection precision and recall, treating both poisoned and drifted clients as positives.
- The accuracy cost of the defense under clean conditions.
- Runtime and memory overhead per round.

**Ablations** (at reduced scale where possible):

- [ ] Block gating on vs. off; final check on vs. off; `caga_max_rel_increase`.
- [ ] Sensitivity to `tol_z`, `tol_rel` and probe size.
- [ ] Root set size, from 12% down to FLTrust-scale (about 100 edges), and a root set biased toward one therapeutic class.
- [ ] Fraction of malicious clients (10–40%) and heterogeneity (Dirichlet α).
- [ ] Onset timing and gradual vs. sudden events.

**Generality:**

- [ ] At least one more dataset: a second DDI dataset (DrugBank DDI or TWOSIDES via TDC) or another `ogbl` link-prediction task.

**Compute budget.** The pilot runs took 3.3–4.8 GPU-hours each. A main table of about 8 methods × 4 conditions × 5 seeds is about 160 runs, or 550–750 GPU-hours, before ablations. Run ablations at reduced scale (fewer rounds or clients) and keep full scale for the main tables. Measure a pilot first; `run_experiments.py` reports per-round timing.

## Phase 4: Reproducibility package

- [ ] Keep the raw CSVs, logs and exact config for every run in the paper. The runner already writes these per run.
- [ ] Write one script per table and figure that regenerates it from the raw outputs.
- [ ] Record the commit hash, hardware and runtime with each result.
- [ ] Release the code under the existing GPL-3.0 license. Note that dataset and ATC metadata terms are separate.
- [ ] State the limitations openly, as the docs already do:
  - shared public graph, not private hospital graphs;
  - no differential privacy or secure aggregation;
  - simulation only;
  - routing labels describe gating actions, not validated diagnoses.

## Phase 5: Writing and submission

**Framing and venue,** to be decided by the end of Phase 2:

- Security framing (robust aggregation, DDI as the application): IEEE TIFS, IEEE TDSC.
- ML framing: IEEE TNNLS, Neural Networks.
- Biomedical framing (robust federated DDI prediction): IEEE JBHI, Briefings in Bioinformatics.

**Core figures:**

1. Hits@20 over rounds, with onsets marked, for each condition.
2. A heatmap of per-client verdicts over rounds (the routing evidence).
3. The routing confusion matrix.
4. Ablation bars.
5. A robustness-versus-overhead plot.

**Related work that must be discussed:**

- Fang et al. (USENIX Security 2020): local model poisoning and the LFR/ERR defenses.
- Zeno and Zeno++: server-side loss scoring of updates.
- FLTrust.
- Krum, median and trimmed mean.
- FLAME.
- Federated graph learning and DDI prediction.

## Risks

| Risk | Signal | Response |
| --- | --- | --- |
| Block gating adds nothing | `caga_noblock` matches CAGA | Simplify the method to whole-client gating and say so; the thesis does not depend on it |
| CAGA does not beat LFR or FLTrust | Phase 3 paired confidence intervals include zero | Lead with no-threat-model robustness across all four conditions; otherwise reframe |
| Clean-condition gain is mistaken for defense | Reviewers note CAGA also wins with no attack | Report attacked-minus-clean gaps per method alongside raw scores |
| Adaptive attacks get through | Damage accumulates below tolerance | Report it honestly and bound the damage per round |
| Training instability masks results | Collapses in clean runs | Phase 1 stability gate |
| Compute overrun | Pilot timing | Reduced-scale ablations; prioritize the joint condition |
| Single dataset | Reviewer objection | Second dataset in Phase 3 |
