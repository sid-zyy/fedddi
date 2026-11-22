# Response and action plan: professor's four critiques

Principle: concede what is true, point to what the paper already shows, and add a small number of new experiments that answer each point with numbers. Nothing below reports a result that has not been run. "New" means not yet run.

---

## 1. What if every centre under-reports?

**Honest position.** The paper only tests 4 of 10 hiders, so it does not currently answer this. The question is fair and matters for deployment.

**Why HARBOR might still help.** FLTrust and the other filters judge clients against each other, so they have nothing to work with when all clients share the same bias. HARBOR's correction comes from the server's trusted pairs, not from the clients. The cap (kappa times the clients' step) keeps it from overriding the federation, so its ceiling is probably the server-only model. That is a hypothesis to test, not a claim.

**Experiments (new, DDInter main, DrugBank secondary, 5 fresh seeds each):**
- A1. All 10 clients hide Major pairs at rho in {0.2, 0.4, 0.6, 0.8}. Compare FedAvg, FLTrust, HARBOR, server-only and a central model. Report MDIR and AUROC.
- A2. Heterogeneous reporting: each centre draws its own rho from Uniform(0.2, 0.8). This is closer to real practice.
- A3. Uniform under-reporting of all severities (not Major-specific), to separate "severity-biased" from "random incompleteness".

**Pre-registered outcomes.** If HARBOR tracks server-only and beats FLTrust at every rho, say so. If everything collapses to the server-only level at high rho, report that as a limit: HARBOR cannot create information that no one holds. Either result is publishable and honest.

**Paper change.** New subsection "Systemic under-reporting" and one sentence in the Limitations.

---

## 2. Why focus on omission instead of "real" poisoning?

**Concede.** Omission is not classic poisoning (injecting wrong data). Calling it "poisoning" loosely invites this objection.

**Respond with three points.**
1. The paper already covers the other families. Table `tab:robust` includes fabricated high-risk pairs, strong data poisoning, sign flip, IPM, min-sum and on-off, and the main benchmark includes ALIE. HARBOR leads FLTrust in all of them (Delta MDIR -0.28 to -0.33). The reviewer may have missed this because the abstract and framing lead with hiding. Fix the framing, not the science.
2. Omission is the real-world gap. Update-based defences handle crafted updates. Honest-looking training on incomplete records is what they miss, and it is also what happens in practice (under-reporting, missing records). **Checked in the code: omission is NOT a label flip here.** Clients draw negatives against the full public training graph (`src/training/trainer.py:79`, graph from `src/data/drugbank.py:69`), so a hidden pair is never trained as a negative. It only loses its positive supervision. Say this precisely: it is a weaker, cleaner attack than label flipping, which is why update-based defences cannot see it. Do not claim it is a label flip.
3. Severity-label manipulation cannot be tested as stated. The model is a binary link predictor and severity is never a training target, so a client changing severity labels changes nothing it trains on. Say this plainly. A severity-predicting head is future work.

**Actions (no new experiments needed for the first three):**
- B1. Rename "hiders" to "under-reporting clients" and call the threat "data omission / under-reporting", not "poisoning". Define the threat model as training-data integrity.
- B2. Move a compact all-attacks table (omission, fabrication, label-flip data poisoning, model-update attacks) from `tab:robust` into the main results, so omission is one row among several.
- B3. Add a Limitations sentence on severity-label manipulation, with the reason above.
- B4. Optional, new: severity-targeted fabrication (fake pairs labelled as Low risk) if the code supports it. It only matters if we add a severity-aware head, so skip it unless asked.

---

## 3. Too many assumptions (80%, 4 of 10, 50% enrichment)?

**Concede, then split the three numbers, because they are different kinds of assumption.**
- 80% and 4 of 10 are attack strengths. Attack strength is not an assumption about the world, it is the stress level. Show a sweep so the reader sees the whole curve.
- 50% enrichment (h) is a design choice, not a population property. Natural share is ~0.25 (DDInter 21.8% Major). The paper already shows h=0.25 still beats the filters (Table `tab:ablation`) and h=0.75 is better.

**Already in the paper:** kappa and h sensitivity, local epochs 1/2/5, 80 rounds, server share 2% to 20%, fixed client samples, full scale, drifted honest clients.

**Experiments (new, 5 fresh seeds):**
- C1. Attack grid: m in {1, 2, 4, 6} by rho in {0.2, 0.4, 0.6, 0.8}. Heatmap of Delta MDIR (HARBOR vs FLTrust). This shows where HARBOR helps and where it stops.
- C2. Population heterogeneity: clients' Major share varies (e.g. 10% to 40%) and sample sizes vary (5k to 40k pairs), so centres differ in data volume and case mix.
- C3. Unequal quality: a fraction of honest centres have random missingness (10 to 30%), so honest noise exists next to hiders.

**Paper change.** Replace the single operating point with the C1 heatmap, add a short "Assumptions" paragraph, and state that our clients are simulated, not real hospital records (already in Limitations; move it up).

---

## 4. Is the server dataset reliable? Test 5/10/20% corrupted labels

**Concede.** The paper says "assumes the server's data and labels are correct" and shows no test of it. This is the most important gap, and the professor's proposed experiment is the right one.

**What severity labels touch.** The server's severity labels only drive enrichment (which positives are picked as high-risk). Training targets are still binary links. So label noise should degrade enrichment toward h=0.25 behaviour, which still beats FLTrust. That is a prediction to test. Also test the harder cases where the link data itself is wrong.

**Experiments (new, DDInter main, 5 fresh seeds, corruption in {0, 5, 10, 20}%; evaluation labels stay clean):**
- D1. Severity-label noise: flip x% of the server's risk labels at random (Major to non-Major and the reverse).
- D2. Biased severity noise: downgrade x% of the server's true Major pairs to non-Major. This is the worst case for enrichment.
- D3. Missing server interactions: drop x% of the server's positive pairs (random), then x% of its Major positives (targeted).
- D4. Corrupted links: x% of server positives replaced by false interactions.
- Report HARBOR, FLTrust (same corrupted root), server-only, and FedAvg (unaffected, as the reference). The question is where HARBOR crosses FLTrust or FedAvg.

**On "why trust the server more than the clients".** Answer in the paper:
- The server is a curated reference source (expert-graded, e.g. DDInter or a pharmacovigilance panel), small and auditable, whereas clients are many, large and uncurated. It is trusted because it can be audited, not because it is large.
- The cap means the server cannot override the federation (Proposition `prop:bound`): a wrong server moves the model at most as far as the clients do.
- Add a deployment check: compare the server-only model with the federated model on held-out data before enabling HARBOR (already in the Discussion). Extend it with a server-audit step.

**Paper change.** New subsection "Imperfect server data" with a figure (MDIR vs corruption %), and a revised Limitations sentence with the measured breaking point.

---

## Order of work

1. D1 to D4 (the professor named this one, and it is the sharpest risk). 
2. A1 to A3 (systemic under-reporting).
3. C1 to C3 (assumption sweeps).
4. Writing changes B1 to B3 (no compute needed; do these while runs go).

**Cost.** All experiments reuse the existing notebooks and seeds scheme. Seeds up to 91 are taken by earlier revision runs, so these use 92 to 111.

**Reply to professor (short version).** Thank you. (1) Systemic under-reporting is untested and we will add it. (2) We will reframe omission as an integrity threat, bring the fabrication, label-flip and update attacks we already ran into the main results, and note that severity labels are not a training target. (3) We will replace the single operating point with an attack-strength grid and heterogeneous centres. (4) We will run the 5/10/20% corruption study on the server's data, covering severity noise, missing pairs and false pairs, and report where HARBOR stops helping.

## What is built (2026-11-22)

Code: `corrupt_server_data` and `hide_random` in `src/data/poisoning.py`; new `cpu_pilot.py` options `--server-corrupt`, `--underreport-*`, `--client-size-range`, `--client-high-range`; presets in `scripts/gpu_variant_study.py`; tests in `tests/test_server_corruption.py`. All on DDInter, pilot scale, fresh seeds.

| Notebook (`results/kaggle/`) | Answers | Seeds | Runs |
|---|---|---|---|
| `fedddi_revision2_05_server_labels` | Q4: wrong Major labels (sn), Major labelled moderate (sd), Major pairs missing (smh) at 5/10/20%, plus clean-server reference | 92-96 | 145 |
| `fedddi_revision2_06_server_pairs` | Q4: missing pairs (sm), false pairs (sf) at 5/10/20% | 92-96 | 90 |
| `fedddi_revision2_07_systemic` | Q1: all 10 centres hide 20-80% of Major (sys2-sys8), per-centre share U(0.2, 0.8) (sysU), any-severity 40/80% (sysAll4/8) | 97-101 | 125 |
| `fedddi_revision2_08_grid_m12` / `09_grid_m46` | Q3: hiders 1/2/4/6 by share 20/40/60/80% | 102-106 | 135 + 120 |
| `fedddi_revision2_10_centres` | Q3: centres of 5k-40k pairs and 10-40% Major vs equal centres; honest centres missing 10-30% of records, with and without hiders | 107-111 | 145 |

Design notes:
- FLTrust ignores severity labels, so it runs only with corrupted pairs (smh, sm, sf), not corrupted labels.
- Server-only ignores clients, so it runs once per notebook under drift.
- Systemic under-reporting starts at round 1 (it is a property of the data, not an attack with an onset).
- Heterogeneous centres need fixed client data, so notebook 10 compares them with equal centres on fixed data and the same seeds.
- Caveat to state in the paper: hidden pairs stay in the public message-passing graph (it stands for a shared reference resource). Under truly systemic under-reporting that resource could be incomplete too; we do not model that.

After the runs: `python scripts/summarize_variants.py <folder>` per subfolder, then write the corruption figure (MDIR vs corruption %) and the grid heatmap.

## Results (2026-11-23)

All 760 runs finished (40 rounds each), DDInter, 5 seeds per cell. Data in `results/revision2/`, tables from `scripts/analyze_revision2.py` (`results/revision2/analysis.txt`), figures from `scripts/make_revision2_figures.py` (`paper_tcbb/figures/fig_server_corruption.pdf`, `fig_systemic.pdf`, `fig_attack_grid.pdf`). MDIR = missed Major rate at 95% specificity, lower is safer. "Delta" = paired HARBOR minus FLTrust on the same seed; every Delta below has p < 0.005 (paired t-test, n = 5).

**Q4 Server data (attack: 4 hiders at 80%).** Clean server: HARBOR 0.087, server-only 0.149, FLTrust 0.260, FedAvg 0.330.

| Corruption at 20% | HARBOR | change vs clean | Server only | FLTrust (same root) |
|---|---|---|---|---|
| Major flag flipped (sn) | 0.107 | +0.019 | 0.192 | 0.260 (ignores labels) |
| Major labelled lower (sd) | 0.088 | +0.000 | 0.148 | 0.260 (ignores labels) |
| Major pairs missing (smh) | 0.093 | +0.006 | 0.165 | 0.273 |
| Pairs missing (sm) | 0.093 | +0.005 | 0.168 | 0.259 |
| False pairs (sf) | 0.140 | +0.053 | 0.197 | 0.302 |

- Label errors and missing pairs barely move HARBOR (at most +0.02 at 20%). Downgrading Major labels does nothing measurable, as predicted: it only shrinks the enrichment pool.
- False pairs are the one real weakness: +0.015 at 10%, +0.053 at 20%. Even then HARBOR (0.140) stays well below FLTrust (0.302) and below its own server-only model (0.197).
- HARBOR beats FLTrust at every corruption type and level (Delta -0.15 to -0.19), and beats server-only everywhere, so the clients still add value when the server is wrong.
- No breaking point was reached within 20%. Say "up to 20% corruption" and not more.

**Q1 Systemic under-reporting (all 10 centres, from round 1).** Server-only 0.158.

| All centres hide Major | 0% | 20% | 40% | 60% | 80% | Uneven U(20-80%) | All severities 40% | All severities 80% |
|---|---|---|---|---|---|---|---|---|
| HARBOR | 0.075 | 0.077 | 0.085 | 0.090 | 0.105 | 0.090 | 0.087 | 0.145 |
| FLTrust | 0.199 | 0.208 | 0.237 | 0.272 | 0.357 | 0.254 | 0.295 | 0.665 (AUROC 0.84) |
| FedAvg | 0.249 | 0.260 | 0.297 | 0.329 | 0.438 | 0.325 | 0.351 | 0.549 |

- The pre-registered worry (HARBOR collapses to server-only) did not happen: HARBOR stays below server-only even at 80% everywhere, and degrades by only +0.03 (Major) or +0.07 (all severities, 80%).
- FLTrust has nothing to compare against when everyone is biased; it degrades with the clients and collapses when every centre drops 80% of all records.
- Caveat to state: hidden pairs stay in the shared message-passing graph, which helps every method; under truly systemic under-reporting that resource could also be incomplete.

**Q3 Attack grid (hiders 1/2/4/6 x share 20-80%).** HARBOR MDIR 0.071-0.087 across the whole grid; FLTrust 0.178-0.247. Delta -0.107 (1 hider, 20%) to -0.161 (6 hiders, 80%), growing with attack strength. Honest reading: about -0.11 of the gap is present without any attack (drift only: HARBOR 0.072, FLTrust 0.182), so the attack-specific part is the growth from -0.11 to -0.16. Attack cost at the strongest cell: HARBOR +0.014, FLTrust +0.066, FedAvg +0.104 over their drift-only levels.

**Q3 Centres (fixed client data).**

| Setting | HARBOR | FLTrust | FedAvg | Server only |
|---|---|---|---|---|
| Equal centres, hideHigh4 | 0.081 | 0.224 | 0.270 | 0.153 |
| Unequal centres (5k-40k pairs, 10-40% Major), hideHigh4 | 0.088 | 0.240 | 0.280 | 0.150 |
| Honest centres missing 10-30%, no hiders | 0.076 | 0.198 | 0.245 | 0.153 |
| Honest missingness plus hideHigh4 | 0.089 | 0.246 | 0.307 | - |

- Unequal sizes and case mix cost HARBOR +0.007; the ranking and the gap do not change (Delta -0.15 in both).
- Honest noise next to hiders adds +0.004 for HARBOR and +0.026 for FLTrust.

**What this means for the reply.** All four critiques now have numbers behind them, and none overturns the paper's claim. The one limit to state plainly is false interactions in the server data: at 20% they raise HARBOR's MDIR from 0.087 to 0.140 (the server-only model rises in step, from 0.149 to 0.197). So the server set must be audited for fabricated pairs, not only for wrong severities.
