# Publication plan

The plan for turning FedDDI-CAGA into a journal paper. Tick items off as they are done, and record decisions here so co-authors share one picture.

Last updated: 2026-10-11.

## Thesis

**CAGA handles data drift and poisoning at the same time by responding to each differently.**

Poisoning and drift damage a federated model in different places:

| Event | What is corrupted | Where the harm should enter | CAGA's intended response |
| --- | --- | --- | --- |
| Poisoning | Labels (fabricated or suppressed interactions), sometimes boosted updates | Every block | Scale down or reject the whole client (`scaled`, `rejected`) |
| Feature drift | Node features only; labels are still correct | Mainly the feature projection | Gate only the feature projection and keep the convolution and MLP updates (`phi_gated`) |

FLTrust, Fang et al.'s loss- and error-based rejection, Krum, median and trimmed mean can only weight or drop a whole client. They must either throw away a drifted client's correct label signal or accept its harmful feature signal. **The contribution is block-level routing: CAGA treats a drifted client differently from a poisoned one.** Every experiment below should test that claim.

Secondary contributions, each of which needs its own ablation:

- CAGA searches over combinations of clients rather than scoring each client in isolation. A non-IID honest client can look harmful alone and still help the combined update.
- The final step-size check limits the damage of each round.

## What "handling drift" means

Decided meaning (**A, drift as corruption**): protect the shared model, evaluated with clean features, from drifted clients while keeping the label signal they contribute. CAGA's clean probe is built for this.

Not claimed (**B, drift as a genuine change in the data**): the shared model serving drifted clients well on their shifted features. A fixed clean probe works against B. If we want B, the planned extension is personalization: a client whose feature projection is gated keeps its own local copy (in the style of FedBN or FedPer). That turns gating into adaptation. Decide before Phase 3 whether this extension is in scope.

Metrics follow from A:

- **Primary:** clean test Hits@20.
- **Signal retained:** CAGA vs. CAGA that rejects drifted clients outright.
- **Reported:** drift-region Hits@20, which drifted clients measure on their own shifted features. A drop here is expected under A and must be explained in the paper.

## Where we are

### Evidence so far (pilot runs, 2026-10-06 and 2026-10-07)

Five 100-round joint-condition runs. They used an older protocol: onset at round 60, 400k observations per client, strong poisoning, `caga_tol_rel=0.1`, lr 0.005. Zip 1 used batch size 16384 and older code; zip 2 used batch size 8192. Treat them as diagnostics, not results.

- **Poison detection works.** After onset, both CAGA runs scaled down or rejected poisoned client 0 in every round from 60 to 69. In zip 1 they also caught poisoned client 1 in 90% of rounds from 70 on. Honest clients were fully accepted in at least 90% of rounds.
- **Drift routing did not happen.** Drifted clients 2 and 3 were fully accepted in at least 90% of post-onset rounds. Their feature projection was gated in 1 of 26 client-rounds. The core claim is so far untested or failing.
- **Only one run held up.** In zip 1, CAGA kept test Hits@20 at 0.10 in rounds 81–100, while both FedAvg seeds collapsed to 0. That is one seed, with Hits@20 swinging between 0.01 and 0.21 from round to round.
- **Training is unstable without any attack.** Norm clipping diverged at round 40, before onset, and zip 2's CAGA had loss spikes before onset. Collapsed runs show a training loss of exactly 2·ln 2 = 1.386: the model predicts 0.5 for every pair and cannot recover.
- **Too much false gating before the attack.** Before onset, some honest clients were not fully accepted in 30–70% of rounds.

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

## Phase 2: Prove drift is a problem and routing fixes it (cheap; go or no-go)

- [ ] **Drift hurts.** Compare drift-only FedAvg with clean FedAvg over a few seeds. The current drift is a +2.0 shift on features 0–3 for 25% of nodes, plus a gradual event on features 2–4. If clean test Hits@20 does not drop measurably, strengthen the drift (more feature dimensions, larger shift, more nodes) or narrow the claim.
- [ ] **Routing fires.** In drift-only CAGA runs, drifted clients should be `phi_gated`. Pass criterion, to be set before running: for example, drifted clients have their feature projection gated in at least 50% of post-onset rounds and honest clients in no more than 10%.
- [ ] If routing does not fire, diagnose before scaling up. Options:
  - Weight the probe toward the drifted region's nodes.
  - Try feature-projection gating before whole-client scaling (`gate_order`, sweep order).
  - Use a per-block tolerance.
  - Check whether the probe can detect feature drift at all, given that it uses clean features.
- [ ] Reduce pre-attack false gating of honest non-IID clients, or measure and explain it.
- [ ] Decide on the personalization extension (meaning B above).

**Gate:** do not start Phase 3 until drift measurably hurts FedAvg and CAGA routes drifted clients to `phi_gated` on drift-only runs.

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
| CAGA without block gating | to do | **Isolates the contribution:** shows that block routing is what handles drift |
| CAGA scoring clients individually | to do | Tests the claim about searching over combinations |

**Attacks.** The current attacks are label poisoning plus update scaling. Add:

- [ ] Model-poisoning attacks: "A Little Is Enough"; min-max and min-sum (Shejwalkar & Houmansadr, NDSS 2021).
- [ ] An **adaptive attacker** that knows CAGA and stays just under its tolerance each round. Our own docs say such attacks can accumulate, so measure how much.
- [ ] Drift that harms more than the feature projection, as a stress test of routing.

**Metrics:**

- Clean test Hits@20, the primary metric.
- Drift-region and non-drift-region Hits@20.
- Attack success: mean probability on fabricated edges, and Hits@20 on suppressed edges.
- **A routing confusion matrix** over rounds: poisoned → rejected or scaled, drifted → `phi_gated`, honest → accepted. Report precision and recall.
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
| Routing never separates drift from poisoning | Phase 2 gate fails | Narrow the claim to robust aggregation with block gating, or redesign the probe |
| CAGA does not beat LFR or FLTrust | Phase 3 paired confidence intervals include zero | Lead with the joint condition and the routing analysis; otherwise reframe |
| Adaptive attacks get through | Damage accumulates below tolerance | Report it honestly and bound the damage per round |
| Training instability masks results | Collapses in clean runs | Phase 1 stability gate |
| Compute overrun | Pilot timing | Reduced-scale ablations; prioritize the joint condition |
| Single dataset | Reviewer objection | Second dataset in Phase 3 |
