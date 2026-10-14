# Publication plan

The plan for turning FedDDI-CAGA into a journal paper. Tick items off as they are done, and record decisions here so co-authors share one picture.

**Start here:** the active plan is **"Paper A: safety-critical poisoning and harm-gated aggregation"** below, which lists the environment, file locations and the exact steps. The earlier sections are the evidence it builds on.

Last updated: 2026-10-14 (Paper A steps 1–4 implemented; seeds 5–7 confirmation failed).

## Current verdict and outlook (2026-10-14)

**Update (2026-10-14, after the seeds 5–7 confirmation): the crafted-attack method claim failed on both datasets and is dropped. Paper A (next section) is the plan; the rest of this section is kept as the record of the earlier verdict.** What changes for Paper A:
- Contributions 1, 2 and 4 (threat model, CCH/MDIR/DRR metrics, benchmark) do not depend on CAGA and are unaffected. The step 3 sanity check already showed the premise holds (`hideHigh4`: MDIR +0.068 at unchanged AUROC).
- Contribution 3 (harm-gated CAGA beating FLTrust, LFR and Zeno, pass criterion 3) is now less likely. Plain CAGA only ties FLTrust on DrugBank, excludes 33–47% of honest client-rounds there (47% with no attack, which threatens the "no-attack cost ≤ 0.005" part of criterion 3), and per-client CCH cannot separate four coordinated hiders (step 3 notes). Revised subjective odds that criterion 3 passes: about 25–35% (before: roughly even).
- So the most likely paper is the pre-planned fallback: **threat, metric and benchmark, with CAGA as one of several defenses** (criteria 1, 2 and 4). Its Q1 odds are mostly unchanged, because that version never relied on CAGA winning. The general-attack results become a supporting section: no defense dominates; FLTrust is strongest on DrugBank; CAGA resists A Little Is Enough where FLTrust collapses on ogbl-ddi.
- Step 5 should be time-boxed (about 2 days): fix the false-exclusion rate and test a group-aware harm gate on `hideHigh4` on tuning seeds before spending GPU time on step 6. If it cannot beat FLTrust's validation MDIR on `hideHigh4` there, skip ahead with plain CAGA as a baseline and put the effort into criteria 2 and 4, the adaptive attacker and the writing.

**Earlier provisional verdict (superseded): pursue publication with CAGA as the method, framed around crafted attacks.** It becomes final once the fresh-seed confirmation (seeds 5–7, both datasets) passes the criteria in Phase 2.

**Claim:** CAGA, a clean-probe gating rule for federated drug-interaction prediction, matches state-of-the-art defenses against standard poisoning and drift, and is the only defense tested that resists crafted "A Little Is Enough" attacks. Size- and similarity-based defenses miss them because CAGA judges an update's effect, not its appearance.

**Which defense handles which attack** (ogbl-ddi, held-out seeds 2–4):

| Defense | Large boosted updates (strong poisoning) | Normal-size poisoning (test A) | Crafted normal-looking updates (test B) |
| --- | --- | --- | --- |
| Norm clipping | Recovers | Fails (fake-edge belief 0.28, vs. 0.30 undefended) | Fails |
| FLTrust | Recovers | Stops it (0.20) | Fails (0.061 Hits@20, no better than FedAvg) |
| **CAGA** | Recovers | Stops it, slightly better on targets (0.18) | **Holds (0.097, every seed)** |

**For it:** the test-B win was on every held-out seed, with a large effect (+0.036 Hits@20, about 60% relative) and a mechanism that explains it. DrugBank pointed the same way (+0.015 AUROC on one tuning seed, a tie on the other). CAGA does not lose elsewhere. The setup is publishable: two datasets, a centralized ceiling, separate tuning and reporting seeds, and pre-registered criteria.

**Against it, or what could still sink it:**
- The decisive result rests on 3 seeds of one dataset; earlier wins on seeds 0–1 did not replicate.
- The advantage is narrow: under ordinary attacks CAGA only ties FLTrust, runs slower, and wrongly blocks 13–23% of honest client-rounds.
- No full-scale CAGA runs and no adaptive-attacker test yet.
- **Novelty is the biggest risk.** Fang et al.'s loss- and error-based rejection (LFR/ERR, USENIX Security 2020) and Zeno/Zeno++ also judge updates by their effect on server data. **If LFR also resists A Little Is Enough, the method claim stops being new.** These baselines are untested and are the next priority.

**Journal outlook:**

| Venue type | Examples | Fit now | Fit after the checklist |
| --- | --- | --- | --- |
| Biomedical / health informatics | IEEE JBHI, Briefings in Bioinformatics, J. Biomedical Informatics | Weak | Good, the natural home |
| Machine learning | IEEE TNNLS, Neural Networks, Knowledge-Based Systems | Weak | Moderate |
| Security | IEEE TIFS, TDSC | Not competitive | Hard: needs adaptive attackers, theory, other domains |

Estimate: better-than-even odds that the confirmation passes on at least one dataset. A Q1 biomedical-informatics paper is realistic if it passes **and** CAGA still beats LFR/Zeno on crafted attacks, in roughly 4–8 more weeks. If LFR matches CAGA, the empirical study is still publishable, but probably not at A level.

**A-level checklist:**
- [x] Fresh-seed confirmation (seeds 5–7): **failed** on 2026-10-14 (see Phase 2). Final tables still need 5 or more seeds.
- [ ] **Closest baselines: Fang's LFR/ERR and Zeno**, plus Krum, coordinate median, trimmed mean, ideally FLAME.
- [ ] Adaptive attacker that knows CAGA's rule and stays under its tolerance.
- [ ] Full-scale GPU tables with the learning-rate decay.
- [ ] Why it works: an analysis or controlled experiment showing why effect-based checks catch what similarity-based checks miss.
- [ ] Ablations: warm-up (DrugBank collapses without it), probe size, root fraction, number of attackers, block gating, and the two failed variants.
- [ ] Cost and limitations: runtime, false-alarm rate, the need for clean server data.
- [ ] DrugBank cold-drug split.

**Decision points:** confirmation passes on both datasets → method paper. On one → narrower method paper. On neither → empirical study.

## Paper A: safety-critical poisoning and harm-gated aggregation (decided 2026-10-14)

**This is the active plan.** It supersedes the "CAGA vs crafted attacks" method paper as the headline, and keeps everything done so far as supporting evidence. Read this section first in a new session.

### Why this paper

Working title: **"Hiding dangerous drug interactions: safety-critical poisoning in federated drug–drug interaction prediction, and harm-gated aggregation."**

The problem: a malicious hospital in a drug-safety federation poisons the shared model so it stops flagging *dangerous* interactions (bleeding, QTc prolongation, serotonin syndrome, CNS depression). Overall accuracy barely moves, so accuracy-based and similarity-based defenses miss it. Test A already showed this effect: overall Hits@20 unchanged while targeted edges were damaged.

Contributions, each able to stand alone:

1. **Threat model:** selective hiding (and fabrication) of high-risk interactions in federated DDI prediction. The literature search (2026-10-14) found no poisoning defense for federated DDI prediction and no work on attacks that hide dangerous interactions.
2. **Metric:** Counterfactual Clinical Harm (CCH), a client-level, effect-based trust score in clinical units, plus the model-level Missed Dangerous Interaction Rate (MDIR) and Damage Recovery Ratio (DRR).
3. **Defense:** harm-gated CAGA, which gates updates by CCH against a risk-weighted, secretly re-drawn clean anchor.
4. **Benchmark:** two datasets, a full attack table, and nine defenses including the closest prior work (Fang et al.'s LFR/ERR, Zeno++, FLTrust).

Why this is the best chance (assessed 2026-10-14): it degrades gracefully. If the defense only ties LFR or FLTrust, contributions 1, 2 and 4 still carry a Q1 biomedical-informatics paper. A paper on "CAGA beats crafted attacks" alone depends on one result and collides with LFR/Zeno. Estimated Q1 acceptance: A about 50–60%, crafted-attack paper about 30–40%, empirical study about 25–35% (subjective). Target venues: IEEE JBHI, Briefings in Bioinformatics, Journal of Biomedical Informatics. A clinical or pharmacology co-author to check the risk mapping would strengthen it more than any single experiment.

### Prior work to cite and position against (from the 2026-10-14 search)

- Validation-based defenses: [Fang et al., LFR/ERR, USENIX Security 2020](https://arxiv.org/pdf/1911.11815); [Zeno++, ICML 2020](https://proceedings.mlr.press/v119/xie20c.html); [FLTrust, NDSS 2021](https://arxiv.org/pdf/2012.13995); [FedVal, USENIX Security 2023](https://arxiv.org/pdf/2306.04040); [FLShield](https://arxiv.org/pdf/2308.05832); [SLVR](https://arxiv.org/pdf/2502.08055). All score updates on overall loss or similarity; none uses a severity-stratified objective or counts harm in clinical units.
- Attacks: [A Little Is Enough](https://arxiv.org/pdf/1902.06156); min-max/min-sum (Shejwalkar & Houmansadr, NDSS 2021); inner-product manipulation (Xie et al., 2020); model replacement (Bagdasaryan et al., 2020).
- Graphs and link prediction: [federated KG-embedding poisoning, Zhou et al., WWW 2024](https://github.com/jisooma/FKGEPoison) (attack, no real defense); [Link-Backdoor](https://arxiv.org/pdf/2208.06776); [poisoning-based link inference in federated GNNs, PoPETs 2026](https://www.petsymposium.org/popets/2026/popets-2026-0042.php).
- Metrics: trust and reputation scores ([FLARE](https://arxiv.org/pdf/2511.14715), [FederatedTrust](https://arxiv.org/pdf/2302.09844)) measure similarity or consistency, not effect. [MPIB's Clinical Harm Event Rate](https://arxiv.org/html/2602.06268v1) is for medical LLM prompts; its finding that attack success and clinical harm diverge supports CCH.
- Certified robustness, for related work only: [FLCert](https://ar5iv.arxiv.org/html/2210.00584).
- Federated drug work: [federated drug–target interaction benchmark](https://arxiv.org/pdf/2302.07684); [DDI ML review](https://www.sciencedirect.com/science/article/pii/S2452310025000113).

### Attack scope

| Family | Attack | Status | Notes |
| --- | --- | --- | --- |
| Data | Interaction suppression / fabrication | done (moderate, strong, test A) | |
| Data | **High-risk hiding** (new, step 3) | implemented (`hideHigh`, `hideHigh4`, `hideHighLIE`); sanity check below | headline attack |
| Data | High-risk fabrication (false alarms) | implemented (`fabHigh`) | secondary |
| Data | Feature drift | done (severe drift) | |
| Model | Boosting / model replacement | done (strong) | |
| Model | A Little Is Enough | done (test B) | CAGA's existing win |
| Model | Sign flip, Gaussian noise | to do (step 7) | sanity rows |
| Model | Min-max / min-sum, inner-product manipulation | to do (step 7) | |
| Model | **Adaptive attacker vs harm gate** | to do (step 8) | mandatory for reviewers |
| System | Malicious majority (6 of 10 clients) | to do (step 8) | |
| System | On-off / sleeper attacker | to do (step 8) | |
| Out of scope | Edge-trigger backdoors; privacy attacks | limitation | say so in the paper |

### Metric definitions (fix before running)

- **Risk classes:** each DrugBank interaction is mapped from its description template to a type (396 templates; the top 20 cover 87% of pairs) and then to a risk class: `high` (CNS depression, QTc prolongation, bleeding, serotonin syndrome, hyperkalemia, nephrotoxicity, hepatotoxicity, cardiotoxicity, arrhythmia, respiratory depression, seizures, rhabdomyolysis, hypoglycemia, hypotension, methemoglobinemia, tachycardia), `moderate` (serum level, metabolism or excretion changes, efficacy changes), `low` (everything else). The mapping lives in a version-controlled file and is reported in full in the paper. About 20% of pairs are high-risk.
- **Decision threshold** t: chosen once per run on the *clean validation* negatives so that 95% of non-interacting pairs score below it (95% specificity). A pair is "flagged" if its probability is ≥ t.
- **MDIR** (model level, lower is better): the fraction of true high-risk test interactions that are not flagged. Also report it per risk class, and the risk-weighted recall Σ_r w_r · recall_r / Σ_r w_r with weights high 3, moderate 2, low 1.
- **CCH_i** (client level, per round, lower is better): on the clean anchor's high-risk positive pairs P_h, let A = the aggregate including client i and A₋ᵢ = the aggregate without it. Then CCH_i = |{p ∈ P_h : flagged under A₋ᵢ, not flagged under A}| + λ · |{n ∈ N_h : not flagged under A₋ᵢ, flagged under A}|, where N_h are sampled non-interacting pairs and λ = 0.5. The unit is "high-risk interactions hidden (plus half of false alarms raised)". Report it per round, and its time mean per client.
- **DRR** (defense level): (score of defense under attack − FedAvg under attack) / (FedAvg clean − FedAvg under attack), for AUROC and for 1 − MDIR. Only reported when the attack costs FedAvg at least 0.005 AUROC or 0.02 MDIR, otherwise it is undefined.
- Always reported alongside: honest-client false-exclusion rate (honest client-rounds not fully accepted) and no-attack cost (defense minus FedAvg, clean).

### Environment and where things are (needed after clearing context)

- Repository: this folder, a git repo with remote `https://github.com/sid-zyy/fedddi` (private). **Several local commits are unpushed; ask the user before pushing.** Never add Claude as author or co-author.
- **DrugBank (licensed, never commit or share):** XML at `C:\Users\Kenzyy\Downloads\drugbank_all_full_database.xml\full database.xml` (5.1, exported 2026-03-05). Processed output in the git-ignored `data/drugbank/`: `drugbank_ddi.pt` (approved small molecules: 2,277 drugs, 674,213 pairs, Morgan fingerprints of 1024 bits, ATC classes, random and cold-drug splits), `drugs.csv`, and `interactions.csv.gz` (pairs with description text). Rebuild: `python scripts/prepare_drugbank.py --xml "<xml path>" --out data/drugbank --approved-only`.
- ogbl-ddi is in `data/` (downloaded by `load_dataset`), with the ATC mapping in `data/ogbl_ddi/atc_mapping.pt`, and fixed-feature caches for seeds 0–4, 10 and 11.
- Python environment: the earlier venv lived in a temporary scratch folder and may be gone. Recreate one with `python -m venv .venv`, then `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install torch-geometric ogb omegaconf hydra-core scipy flwr wandb scikit-learn networkx tqdm rdkit lxml pandas pytest pyyaml`. `.venv/` is git-ignored.
- Tests: run each file separately (legacy test doubles break shared processes): `for t in tests/test_*.py; do python -m pytest "$t" -q -p no:cacheprovider | tail -1; done`. All 180+ pass as of 2026-10-14 (recreated `.venv` with torch 2.14.1+cpu, torch-geometric 2.8.1, flwr 1.39.0).
- Pilot runner: `scripts/cpu_pilot.py` (one federated run; `--dataset {ogbl-ddi,drugbank} --split {random,cold} --method {fedavg,fedavg_root,normclip,fltrust,caga,caga_noblock} --cond {clean,poison,drift,joint} --device {cpu,cuda}`; attack options `--poison-strength`, `--extra-poisoned`, `--no-boost`, `--lie-clients`, `--lie-z`; CAGA options `--caga-warmup`, `--caga-tol-z`, `--caga-class-aware`, `--caga-reputation`). It writes `rounds.csv` with per-round metrics, per-client losses and CAGA verdicts.
- Study runner: `scripts/gpu_variant_study.py` (parallel, resumable, `--methods`, `--attacks`, `--seeds`); summary: `scripts/summarize_variants.py <folder>`.
- Kaggle: `python scripts/make_kaggle_bundle.py` writes `results/kaggle/fedddi_code.zip` and the notebooks. The user updates the Kaggle dataset `fedddi-phase-3` with the zip, and attaches the **private** DrugBank dataset holding `drugbank_ddi.pt`. A T4 runs about 3 pilot runs in parallel, about 7 minutes each on DrugBank. Notebooks: use `!` cells for long runs (live output); `%%bash` cells only print when they finish.
- Laptop: CPU runs pause when it sleeps and die if the battery runs out; remind the user to keep it plugged in.
- Status replies: when the user asks "status", answer only with runs done, health, and ETA.

### Pending from before this decision

- [x] (2026-10-14: done, the crafted-attack claim **fails**; see Phase 2, "Confirmation on fresh seeds 5–7") Kaggle `fedddi_confirm_drugbank.ipynb` (84 runs, seeds 5–7) and `fedddi_confirm_ogbl.ipynb` (60 runs) for the crafted-attack claim. Its first attempt failed because the DrugBank dataset was not attached. Their results become the "general attacks" section of Paper A; evaluate them against the criteria in Phase 2.
- [ ] Kaggle `fedddi_lrdecay.ipynb` and `fedddi_centralized.ipynb` (methods section: learning-rate decay and centralized ceilings).

### Exact steps

**Step 1: risk labels (about 1 day).**
1. ✅ Add `conf/ddi_risk_map.yaml`: an ordered list of `{pattern, type, risk}` rules matched against the template (description with both drug names replaced by `#DRUG`). Cover the top 60 templates explicitly, then keyword fallbacks; everything unmatched is `low`.
2. ✅ In `scripts/prepare_drugbank.py`: build the template per pair (the code exists in the 2026-10-14 analysis: replace each drug's name with `#DRUG`), map it to `type_id` and `risk` (0 low, 1 moderate, 2 high), and save `edge_type` and `edge_risk` aligned with `edges`, plus a `risk` tensor for every split's positive edges (`splits[s][part]["risk"]`). Write `data/drugbank/templates.csv` (template, count, type, risk) for the paper's appendix.
3. ✅ In `src/data/drugbank.py`, return the per-split risk tensors (keep the current return signature backward compatible, e.g. add `split[part]["risk"]`).
4. ✅ Tests in `tests/test_drugbank.py`: mapping rules on synthetic descriptions; risk arrays aligned with edges; the loader returns risk.
5. ✅ Rebuild `data/drugbank/drugbank_ddi.pt`. Print the class counts (expect about 20% high) and save them in this plan.

**Step 1 result (2026-10-14).** Labelled with `python scripts/prepare_drugbank.py --risk-only --out data/drugbank` (369 distinct templates; 82 rules: the top 60 templates explicitly, then 22 keyword fallbacks). Class counts over all 674,213 pairs: **high 171,480 (25.4%)**, moderate 414,870 (61.5%), low 87,863 (13.0%). Largest high-risk types: CNS depression 54,495, QTc prolongation 32,369, hypotension 15,845, methemoglobinemia 15,115, bleeding 12,331, hyperkalemia 9,251, nephrotoxicity 6,926, arrhythmia 6,560, tachycardia 5,536, hypoglycemia 4,641, serotonin syndrome 4,404, rhabdomyolysis 2,578 (others under 600). High-risk share per split: random 0.254 / 0.254 / 0.255 (train / valid / test); cold 0.256 / 0.229 / 0.269. The per-template table is in the git-ignored `data/drugbank/templates.csv`.

Decisions and reasons:
- **High is 25.4%, not the expected ~20%. Kept as is.** Every high-risk pair falls in one of the 16 classes fixed in advance; trimming classes to reach a target share would be choosing the label set by its outcome. The paper reports the share as measured.
- **Relabelled the existing build instead of re-parsing the XML.** `--risk-only` reads `drugs.csv` and `interactions.csv.gz` and adds the labels in place; the splits, features and edges were checked to be byte-identical to the previous file, so every earlier DrugBank result (tuning seeds 10–11, the Kaggle confirmation) stays valid. A full rebuild from the XML labels the same way.
- **Judgement calls in the mapping** (all listed at the top of `conf/ddi_risk_map.yaml`, for the clinical co-author to check): increased anticoagulant or antiplatelet activity counts as bleeding; bradycardia, heart block and AV block as arrhythmia; renal failure as nephrotoxicity; "serotonergic activities" as serotonin syndrome; plain sedation without "CNS depression" stays low; generic "adverse effects" and hypertension stay low (not in the fixed list); decreases of a toxic activity and "risk … can be decreased" are low; decreases of a therapeutic activity are efficacy changes (moderate).
- **Drug synonyms.** DrugBank sometimes names a drug differently in the text (Salbutamol for albuterol, Rifampicin, Oxetacaine, Florbetapir/Florbetaben, ONC-201), so a few templates keep a drug name. They are left out of the explicit list and caught by the keyword fallbacks.
- **Licence:** `conf/ddi_risk_map.yaml` quotes the 60 most common DrugBank description templates (drug names removed) and their counts. The user approved committing it (2026-10-14); the per-pair data and `templates.csv` stay git-ignored.

**Step 2: metrics (about 1 day).**
1. ✅ New `src/training/safety_metrics.py`: `threshold_at_specificity(neg_scores, 0.95)`, `mdir(pos_scores, pos_risk, t)`, `recall_by_risk(...)`, `risk_weighted_recall(...)`, `cch(scores_with, scores_without, risk, neg_with, neg_without, t, lam=0.5)`, `drr(def_attack, fedavg_attack, fedavg_clean)`.
2. ✅ `scripts/cpu_pilot.py`: after each round's evaluation, compute t on validation negatives and log `test_mdir_high`, `test_recall_low/moderate/high`, `test_risk_weighted_recall`, `threshold`. With `--log-cch`, compute CCH for every client every round (leave-one-out on the clean anchor, for any method, so the metric is comparable across defenses) and log `cch` as a JSON list.
3. ✅ `scripts/summarize_variants.py`: add these columns, plus DRR against FedAvg on the same seed and attack, and the honest false-exclusion rate.
4. ✅ Unit tests with hand-computed toy scores.

**Step 2 notes (2026-10-14).** Every DrugBank round now logs `threshold` (95% specificity on the clean validation negatives, recomputed each round for that round's model), `val_/test_mdir_high`, `val_/test_recall_{low,moderate,high}` and `val_/test_risk_weighted_recall`; validation versions exist because step 6 tunes on validation MDIR. Every run also logs `excluded`, a per-client flag for "not fully accepted", defined per defense: clipped (norm clipping), trust ≤ 0 (FLTrust), verdict other than "accepted" (CAGA), dropped (LFR, ERR, Zeno, not chosen by Krum), never (FedAvg, median, trimmed mean). The summarizer turns it into the honest false-exclusion rate (honest client-rounds from round 22 on).

Decisions and reasons for `--log-cch`:
- **Anchor:** a fixed, seeded sample of 2,000 validation high-risk positives and 2,000 validation negatives, the same for every defense. Reason: the logged CCH must be comparable across defenses, and most baselines have no server data. Harm-gated CAGA (step 5) will compute its own CCH on the server's secret root probe; the validation anchor is for evaluation only.
- **A₋ᵢ is the defense's own aggregate without client i** (the aggregation is re-run on the other nine clients, with CAGA's reputation state copied so the re-runs do not advance it). Reason: CCH should measure the harm a client does *through that defense*.
- **One threshold for A and A₋ᵢ**: the deployed aggregate A's 95%-specificity threshold on the anchor negatives. Reason: harm is judged at the operating point actually deployed; giving each leave-one-out model its own threshold would hold its false-alarm count at 5% by construction and empty the λ term.

**Step 3: safety-critical attacks (about 1 day).**
1. ✅ `src/data/poisoning.py`: `PoisonEvent` gains `hide_risk: float` (the fraction of the client's own high-risk positives to drop) and `fabricate_high: float` (fake pairs drawn among drug pairs whose ATC-class combination often carries high-risk interactions). `PoisoningSimulator` receives an `edge_risk` lookup keyed by `(min(u,v), max(u,v))`. Fabricated and hidden pairs are recorded so both metrics can be computed.
2. ✅ `scripts/cpu_pilot.py` flags: `--hide-high FRAC`, `--fabricate-high FRAC`; attack presets in `scripts/gpu_variant_study.py`: `hideHigh` (2 attackers, frac 0.8, normal-size updates, severe drift), `hideHigh4` (4 attackers), `fabHigh`, `hideHighLIE` (the hidden pairs plus A Little Is Enough).
3. ✅ Tests: the hidden pairs come only from the high-risk class; the counts match the fraction.
4. ✅ (passes with `hideHigh4`, see below) Sanity check: under `hideHigh`, FedAvg's MDIR must rise by at least 0.05 while AUROC moves by less than 0.01. If not, raise the fraction or the attacker count before going on; the paper's premise depends on it.

**Step 3 notes (2026-10-14).** `--hide-high` and `--fabricate-high` keep the protocol's attacker clients and timing (client 0 sudden at round 12, client 1 gradual from round 22) and replace their class-based suppress/emerge attack. Presets: `hideHigh` = joint (severe drift on clients 2–3) + hide 0.8 + normal-size updates; `hideHigh4` adds clients 4 and 5; `fabHigh` = fabricate 0.3; `hideHighLIE` = clients 0–1 hide and clients 4–5 run A Little Is Enough; `drift` = drift only (the control). Decisions and reasons:
- **Hiding drops the pairs from the attacker's positives**, as specified; they remain in the shared message-passing graph, which the pilot keeps global. This is the weaker, plan-specified form; if the sanity check fails, the next strengthening to consider (after the fraction and attacker count the plan names) is reporting the hidden pairs as explicit negatives.
- **Fabrication** draws pairs from ATC class pairs whose high-risk share is at least twice the overall share (class pairs with ≥ 100 interactions), weighted by their number of high-risk interactions, and never uses a known interaction. The fraction is relative to the client's observations, like `emerge`; 0.3 matches the moderate attack's emerge strength.
- **`hideHighLIE` uses different clients for the two parts** because A Little Is Enough overwrites the attacker's whole update, so a client cannot both hide pairs and send a crafted update.
- **Sanity-check comparison, fixed before seeing results:** the primary comparison is FedAvg under `hideHigh` against FedAvg under `drift` (same severe drift, no hiding), because `hideHigh` includes the drift and the premise is about the hiding; FedAvg `clean` is reported too. Metrics: test MDIR and test AUROC, last-10-round window, mean over tuning seeds 10 and 11 (validation numbers reported alongside). Pass: MDIR up ≥ 0.05 and |ΔAUROC| < 0.01.

**Step 3 sanity check, first attempt (2026-10-14, CPU, `results/paperA_sanity`, FedAvg, DrugBank random split, tuning seeds 10–11, last-10-round means): fails.**

| Condition | Test AUROC | Test MDIR (high) | Val MDIR | Recall high / moderate / low | Mean CCH poisoned / drifted / honest |
| --- | --- | --- | --- | --- | --- |
| clean | 0.9305 | 0.4965 | 0.4978 | 0.50 / 0.69 / 0.55 | – / – / 19.6 |
| drift only | 0.9282 | 0.4708 | 0.4733 | 0.53 / 0.66 / 0.54 | – / 56.4 / 19.0 |
| `hideHigh` | 0.9286 | 0.5077 | 0.5088 | 0.49 / 0.68 / 0.55 | 24.1 / 53.1 / 18.5 |

- Against drift only (the pre-registered comparison): **MDIR +0.037** (seed 10 +0.024, seed 11 +0.050), AUROC +0.0004. The AUROC half passes; the MDIR half (≥ 0.05) does not. Against clean: MDIR +0.011, AUROC −0.002.
- The effect is real but small: the attack targets high-risk pairs specifically (high-risk recall drops 0.03–0.05 while moderate and low barely move, and AUROC does not change), which is the premise's shape, but two attackers dropping 80% of their own high-risk pairs is not enough. The model still gives the hidden pairs a mean probability of 0.66.
- Drift alone *lowered* MDIR against clean (−0.026), about the size of the seed-to-seed spread. So two seeds cannot separate effects much below ~0.03, and "against clean" is a noisy reference here.
- CCH already ranks the hiders above honest clients (24 vs 18.5 per round), but the severely drifted clients score far higher (53). A harm gate built on CCH would act on drifted clients first. Keep this in mind for step 5 and the paper (it is arguably correct: those clients do hide the most high-risk pairs).
- Baseline MDIR is about 0.5 at 95% specificity: the scaled-down pilot model misses half of the high-risk interactions even when clean. MDIR should be lower at full scale (step 10).

**Decision (2026-10-14): strengthen by attacker count first, using the planned `hideHigh4` preset (4 attackers, the same fraction of 0.8), on seeds 10–11.** Reason: the fraction can only go from 0.8 to 1.0, which would give at most about 0.046 if the effect is linear, still short; doubling the attackers is the larger lever and is already in the attack table. If `hideHigh4` passes, it becomes the headline attack for criterion 1, and `hideHigh` stays as the weaker row. If it does not, the next step is reporting hidden pairs as explicit negatives (a stronger and realistic form: the hospital records them as "no interaction"), which needs a small trainer change.

**Step 3 sanity check, second attempt (2026-10-14, `hideHigh4`, same setup): passes.**

| Condition | Test AUROC | Test MDIR | Val MDIR | Recall high | Mean CCH poisoned / drifted / honest |
| --- | --- | --- | --- | --- | --- |
| drift only | 0.9282 | 0.4708 | 0.4733 | 0.53 | – / 56.4 / 19.0 |
| `hideHigh` | 0.9286 | 0.5077 | 0.5088 | 0.49 | 24.1 / 53.1 / 18.5 |
| `hideHigh4` | 0.9259 | **0.5391** | 0.5412 | 0.46 | 21.0 / 46.5 / 19.2 |

- Against drift only: **MDIR +0.068** (seed 10 +0.054, seed 11 +0.083, both above 0.05), **AUROC −0.002** (validation: +0.068 and −0.003). Criterion 1 passes on both seeds and on both validation and test.
- Against clean the rise is +0.043 MDIR with −0.005 AUROC, under 0.05. The pre-registered comparison is drift only, so this does not change the verdict, but the paper must report both and say why drift only is the right reference (the attack runs with the same drift).
- **Decision: `hideHigh4` is the headline attack for criterion 1 and the main tables; `hideHigh` stays as the weaker row** (it shows the dose response: 2 attackers +0.037, 4 attackers +0.068). The explicit-negatives variant is not needed now.
- **Warning for step 5: with 4 hiders, CCH no longer separates them from honest clients (21.0 vs 19.2 per round).** Leave-one-out removes one attacker at a time, and the other three still pull the model the same way, so each attacker's marginal harm looks small (the attackers mask each other). Drifted clients still score highest. A harm gate that tests clients one at a time will likely miss coordinated hiders. Before tuning, step 5 should consider a group-aware variant (for example, scoring each client against an aggregate of only the clients CAGA already accepts, or leaving out clusters of similar updates) and test it on `hideHigh4`.

**Step 4: baselines (about 1–2 days).** In `src/training/robust_aggregation.py`, numpy only, using the same clean probe as CAGA:
1. ✅ `lfr_aggregate`: for each client, the probe loss of the aggregate without it; remove the c clients whose removal lowers the loss most; average the rest. c = the true number of attackers (Fang et al.'s assumption, favourable to LFR).
2. ✅ `err_aggregate`: the same with probe ranking error (1 − AUROC on the probe).
3. ✅ `zeno_aggregate`: score_i = L(θ) − L(θ + Δᵢ) − ρ‖Δᵢ‖², keep the top n − b, average them (ρ = 0.0005, b = c).
4. ✅ `krum_aggregate`, `median_aggregate`, `trimmed_mean_aggregate` (trim 20%).
5. ✅ Wire them as `--method {lfr,err,zeno,krum,median,trimmed}` in the pilot and the study runner. Tests on toy updates for each.

**Step 4 notes (2026-10-14).** All six are in `src/training/robust_aggregation.py` with toy tests, and run as `--method` in the pilot and the study runner. LFR, ERR and Zeno score on the same per-round root probe as CAGA (so they reserve root edges like CAGA and FLTrust). Decisions and reasons:
- **c, b and f = the true number of attackers** (poisoners plus A Little Is Enough clients), as the plan says, **but 2 when a run has no attackers** (`--byz-c` overrides). Reason: with c = 0 the clean and drift-only runs would collapse to FedAvg and hide each defense's no-attack cost; a deployed defense does not know there is no attack, and 2 is the smallest attacker count in the study.
- **ERR** uses 1 − AUROC on the probe; **Krum** is single Krum (one update chosen); **trimmed mean** drops 20% from each end per coordinate (2 of 10 clients each side). Median and trimmed mean are unweighted, as in their papers.

**Step 5: harm-gated CAGA (about 2 days).**
1. ✅ `CAGAHyperparams`: `harm_gate: bool`, `harm_tol: float` (in high-risk interactions per round, as a fraction of |P_h|), `risk_weights: (1, 2, 3)`, `probe_high_frac: 0.5` (the probe oversamples high-risk positives to this share), `secret_probe: bool`.
2. ✅ `caga_round`: when `harm_gate`, compute CCH_i for every client with the leave-one-out the reputation code already does. Only clients with CCH_i > harm_tol are eligible for shrinking or rejection, and moves are judged on the risk-weighted per-sample loss (`sample_weights` = risk weight of each probe positive, the same for every client). The final step-size check also uses the risk-weighted loss. Report `caga_cch` per client.
3. ✅ Probe sampling (`sample_probe_edges`): stratify positives by risk to reach `probe_high_frac`; with `secret_probe`, seed with `hash(server_secret, round)` instead of the round number.
4. ✅ Pilot flags `--caga-harm-gate --caga-harm-tol X --caga-secret-probe`; study method `caga_harm`.
5. ✅ Tests: a client hiding high-risk pairs is gated while overall loss barely changes; an honest client is not; the tolerance is respected.

**Step 5a: CAGA's false alarms (2026-10-14).** Diagnosis from the seeds 5–7 DrugBank clean runs (no attack): about 5 of 10 honest clients are gated per round (half of them rejected outright), the *same* clients round after round (on seed 5, three clients in 97% of rounds, one in 6%), and it does not fade with training. Across the 30 clients, a client's gating rate rises with how far its drug-class mix is from the global mix (total-variation distance; Spearman 0.66, p < 0.0001). Cause: the paired test asks whether removing a client helps the probe *with certainty*; with ~2,000 paired probe samples any real difference is significant, and a specialist's update always looks slightly worse on a probe drawn from the global mix. Earlier fixes failed for this reason: `tol_z` 3 attacks noise, which is not the problem; reputation averages the same biased score; the class-aware probe removes the bias but is exploitable.

Fix (implemented 2026-10-14, `CAGAHyperparams.outlier_gate`, `outlier_k`, `baseline_gate`; pilot `--caga-outlier --caga-outlier-k --caga-baseline`; study methods `caga_out`, `caga_cal`; tests in `tests/test_caga.py`): only clients whose leave-one-out effect is a robust outlier *across clients* in that round (robust z = (effect − median) / (1.4826·MAD) > k = 3) may be gated; with `baseline_gate`, each client's mean score over the 5 warm-up rounds is subtracted first (frozen afterwards), which cancels a specialist's steady offset. Known limitation: an attacker active from round 1 sets its own baseline (the warm-up already assumes no early attack); `caga_out` without the baseline covers that case.

Test, criteria fixed before running (CPU, DrugBank random, tuning seeds 10–11, `results/paperA_cal`): `caga` (current) and `caga_cal` under `clean`, `moderate`, `testB`, `hideHigh4`; `caga_out` under `clean` and `testB`. `caga_cal` is adopted as CAGA's default if (i) honest false exclusion (from round 22) under `clean` is ≤ 0.10; (ii) under `testB` and `moderate` the attackers are still not fully accepted in ≥ 80% of post-onset rounds; (iii) mean validation AUROC is no more than 0.003 below `caga` under every condition. Under `hideHigh4`, the attackers' exclusion rate and validation MDIR are reported (they feed step 5b). If (i) passes but (ii) fails, try k = 2; if (i) fails, compare with `caga_out` to see which part is not working.

**Step 5b: harm-gated CAGA, implemented (2026-10-14).** `CAGAHyperparams.harm_gate`, `harm_tol`, `harm_mode`, `harm_lambda`, `risk_weights`; `caga_round(..., score_fn, probe_risk)`; `sample_risk_probe` (stratified to `probe_high_frac`) and `probe_seed` (secret hash) in `src/training/caga_probe.py`; pilot `--caga-harm-gate --caga-harm-tol --caga-harm-mode --caga-probe-high-frac --caga-secret-probe`; study methods `caga_harm` and `caga_harm_solo`; tests in `tests/test_caga.py` and `tests/test_caga_probe.py`. Decisions and reasons:
- **Two harm modes.** `loo` is the plan's definition (CCH of including the client). `solo` (added) compares the client's own model with the full aggregate. Reason: in the step 3 runs, leave-one-out CCH could not separate four coordinated hiders, and the toy test reproduces it exactly (four hiders: every client's leave-one-out CCH is 0; solo CCH flags all four and no honest client). Solo harm also penalises honest specialists (their own model misses the classes they lack), so `caga_harm_solo` adds the step 5a calibration: the outlier score and warm-up baseline are computed on the harm values.
- **Threshold:** the full aggregate's 95%-specificity threshold on the probe negatives, used for every comparison in the round (same reasoning as for the logged CCH).
- **Eligibility combines with AND**: a client may be gated only if it passes every enabled condition (harm above tolerance; outlier score above k if enabled).
- **Risk weights also rescale the final step-size check** (weighted mean loss), as specified.
- **Secret probe:** seeded by SHA-256 of a per-run server secret and the round number.
- **Open:** whether either mode works on real data; that is step 6. If `caga_cal` passes step 5a, the step 6 grid uses it as the base for `caga_harm` too.

**Step 6: tuning (DrugBank random split, seeds 10–11, validation only).** GPU via a new Kaggle notebook (add it to `scripts/make_kaggle_bundle.py`):
- attacks `hideHigh`, `hideHigh4`, `fabHigh`, `moderate`, `testB`;
- methods `fedavg`, `normclip`, `fltrust`, `lfr`, `zeno`, `median`, `caga`, `caga_harm` with harm_tol ∈ {0.005, 0.01, 0.02} and probe_high_frac ∈ {0.3, 0.5};
- choose harm_tol and probe_high_frac by validation MDIR, subject to validation AUROC within 0.005 of FLTrust. Record the choice and the reason here before step 7.

**Step 6 set-up (2026-10-14).** Two Kaggle notebooks, built by `scripts/make_kaggle_bundle.py`: `fedddi_paperA_tune_baselines.ipynb` (90 runs: FedAvg, norm clip, FLTrust, LFR, Zeno, median, `caga`, `caga_cal` × the five attacks × seeds 10–11) and `fedddi_paperA_tune_harm.ipynb` (120 runs: `harm_{loo,solo}_t{005,01,02}_h{3,5}`). Both need the relabelled `drugbank_ddi.pt` on Kaggle (the notebooks check). Selection rule, fixed before results, extending the one above to the harm mode: pick the (mode, harm_tol, probe_high_frac) with the lowest mean validation MDIR over `hideHigh` and `hideHigh4`, among settings whose mean validation AUROC is within 0.005 of FLTrust's under every one of the five attacks; ties within 0.005 MDIR go to the setting with the lower honest false-exclusion rate. If no setting meets the AUROC constraint, Paper A goes ahead with the threat/metric/benchmark framing (criteria 1, 2, 4) and harm-gated CAGA is reported as a negative result.

**Step 7: main results on fresh seeds 12–16 (5 seeds; never used before).**
- DrugBank random split: every attack in the scope table plus `clean`, every baseline plus `caga` and `caga_harm`.
- DrugBank cold-drug split: `clean`, `hideHigh`, `testB`.
- ogbl-ddi: the general attacks only (no risk labels): `clean`, `strong`, `testA`, `testB`, `minmax`.
- Add `signflip`, `gauss`, `minmax`, `minsum` and `ipm` attack presets first (step 4 style, in the pilot's update-replacement hook next to `lie_updates`).

**Step 8: robustness of the defense itself (seeds 12–16, DrugBank).**
1. Adaptive attacker: knows harm_tol and estimates CCH on its own data; each round it hides as many high-risk pairs as it can while keeping its estimated CCH just under the tolerance. Report cumulative MDIR over rounds, with and without `secret_probe`.
2. Malicious majority: 6 of 10 clients hide high-risk pairs.
3. On-off attacker: attacks in rounds 12–20 and 30–40 only.

**Step 9: ablations (seeds 10–11 are enough).** Risk weighting off; harm gate off (plain CAGA); secret probe off; warm-up 0 (it collapsed on DrugBank before); probe size 1000/4000/8000; root fraction 0.05/0.12/0.2; block gating on; class-aware and reputation variants (negative results).

**Step 10: full scale and cost.** Run the headline table (DrugBank, `clean`, `hideHigh`, `testB`; FedAvg, FLTrust, LFR, harm-gated CAGA; 3 seeds) through `main_fl.py` on Kaggle at full scale (hidden size 500, 200k observations, 100 rounds, `training.round_lr_decay=true` if the lrdecay check passed). This needs the new methods and metrics wired into `main_fl.py`'s strategy. Report runtime per round for every method.

**Step 11: writing (about 1–2 weeks, overlapping with steps 8–10).** Outline:
1. Introduction: patient-safety motivation and the threat.
2. Related work: robust FL, validation-based defenses, graph and KG poisoning, DDI prediction.
3. Threat model.
4. Metrics: CCH, MDIR, DRR.
5. Method: harm-gated CAGA.
6. Setup: datasets, splits, risk mapping, federation, attacks, baselines, seeds.
7. Results: the attack-by-defense table; MDIR vs AUROC; CCH traces; adaptive and majority attacks; cold drugs; ogbl-ddi generality; cost.
8. Ablations.
9. Limitations: template-based risk labels, backdoors and privacy out of scope, simulated federation, clean server data assumption.
10. Conclusion.

Figures: an MDIR-vs-AUROC scatter per defense (shows that accuracy hides harm); per-client CCH heatmap over rounds; cumulative MDIR under the adaptive attacker; DRR bars per attack.

### Pass criteria for Paper A (fixed 2026-10-14, before any results)

1. **Premise:** under `hideHigh`, FedAvg's MDIR rises by ≥ 0.05 with an AUROC change < 0.01 (step 3 sanity check). If this fails, the attack is strengthened before anything else.
2. **Metric value:** at least one standard defense (FLTrust, LFR or Zeno) keeps AUROC within 0.01 of clean while its MDIR stays ≥ 0.03 above clean. This is the "accuracy hides harm" finding, and it carries the paper even if criterion 3 fails.
3. **Defense:** on seeds 12–16, harm-gated CAGA has lower MDIR than FLTrust, LFR and Zeno under `hideHigh` and `hideHigh4`, with the paired 95% interval clear of zero; its AUROC is no more than 0.005 below FLTrust's under every attack; and its no-attack AUROC cost is ≤ 0.005.
4. **Adaptive attacker:** with `secret_probe`, cumulative MDIR stays at least 50% below FedAvg's.

Outcomes: 1–4 pass → full Paper A. Criteria 1, 2 and 4 pass but 3 does not → Paper A framed as threat, metric and benchmark, with CAGA as one of several defenses. Criterion 1 fails even after strengthening → return to the empirical study.

### Timeline

| Week | Work |
| --- | --- |
| 1 | Steps 1–4 (labels, metrics, attacks, baselines) |
| 2 | Step 5 (harm gate) and step 6 (tuning) |
| 3 | Steps 7–8 (main results, adaptive, majority) |
| 4 | Steps 9–10 (ablations, full scale) and writing |
| 5–6 | Writing, figures, internal review, submission |


## Thesis

**Refined 2026-10-14: see "Current verdict and outlook" above for the claim now being tested.** The text below is the Option A thesis it grew from.

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
- [x] **Full-scale GPU check (Kaggle T4, 2026-10-13):** clean FedAvg, 10 clients, 200k observations, 100 rounds, seed 10. Both runs were stable, with no collapse. With 2 local epochs, test Hits@20 peaked at 0.196 (round 27) and fell to 0.116 over rounds 81–100. With 5 epochs it peaked at 0.213 (round 15) and ended at 0.131. The CPU finding that fewer local epochs are better did not carry over, so keep 5. **Both runs peak early and then decline**, so a window at rounds 81–100 misses the best model. The likely cause is that the learning rate never decays across rounds: each client restarts its schedule every round. Next: add a decay across rounds, or report the best-validation round. Runtime: 1.7 h (2 epochs) and 2.2 h (5 epochs) per FedAvg run.
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

  **Decision (2026-10-13): drop block gating; CAGA candidate is whole-client gating with warm-up 5.** With warm-up 5, no-block scored 0.131 / 0.155 (mean 0.143) against 0.140 / 0.134 (0.137) with block gating. Across all 6 paired seed comparisons, block gating minus no-block was +0.029, −0.006, +0.005, −0.020, +0.009, −0.021, a mean of about 0. Without block gating, honest clients are fully accepted more often (92% vs. 79% after onset) and rounds are faster. The trade-off: drifted clients are let through more often (20% vs. 5% of post-onset rounds; poisoned 5% vs. 1%) with no loss of accuracy. Report this in the paper. Block gating stays as an ablation (`caga_gate_blocks=true`). Baseline settings are still being chosen.

- [ ] **CAGA tuning** on separate tuning seeds, validation Hits@20 only: `tol_z`, `tol_rel`, probe size, and possibly a no-gating warm-up. Targets: false gating of honest clients early in training, and the clean-vs-attacked gap.
- [x] **Baseline tuning** (seeds 10–11, validation): norm clip ×1.0 0.151 (best; ×1.5 0.138, ×2.0 0.132, ×3.0 0.136); FLTrust root 10k 0.119, 20k 0.113, 40k 0.072; FedAvg 0.126. On these seeds **norm clip ×1.0 tied CAGA** (0.143), and the joint attack did not lower FedAvg's validation score (0.126 clean and joint), so the defenses could not be separated. The high FLTrust training loss came from drifted clients (1.61 against 0.41 for honest clients on seed 11); the averages up to 8 on seed 10 were a one-seed spike.
- [x] **Frozen settings (2026-10-13), CPU scale:** lr 0.003, 2 local epochs; CAGA whole-client gating with warm-up 5; norm clip ×1.0; FLTrust root 20k (tied with 10k, more consistent, and matches a client's step count).
- [x] **Held-out confirmation (2026-10-13), seeds 2–4, test Hits@20, 36 runs.** Strong poisoning: FedAvg 0.026, norm clip 0.109, FLTrust 0.111, CAGA 0.104 (CAGA − FedAvg +0.078, 95% interval +0.056 to +0.101). Moderate poisoning: FedAvg 0.100, norm clip 0.113, FLTrust 0.111, root client 0.118, CAGA 0.119. Clean: FedAvg 0.091, norm clip 0.087, CAGA 0.098. CAGA vs norm clip and vs FLTrust: every interval includes zero. CAGA trusts fake edges less than norm clip on all 6 seed–condition pairs (by 0.017–0.026); FLTrust protects hidden real edges better than CAGA on all 6. CAGA fully accepts honest clients in only 74–84% of rounds. Root client beat FedAvg under moderate poisoning (+0.018, interval +0.004 to +0.030).
- **Verdict (2026-10-13):** CAGA works as a defense but is not better than norm clipping or FLTrust on held-out seeds. Do not submit as a new, better method yet. Next: unboosted attacks with 3–4 attackers, "A Little Is Enough", an adaptive attacker, and an attack from round 1. If CAGA then beats both baselines with intervals clear of zero, run the full GPU protocol and write a method paper; otherwise write an empirical study. Full write-up: https://claude.ai/code/artifact/7640891d-9b34-43b4-82d0-a629f30ad4cd
- [ ] **Decisive attack tests (running, 2026-10-13):** frozen settings, held-out seeds 2–4, FedAvg / norm clip ×1.0 / FLTrust 20k / CAGA. **A:** 4 of 10 clients poisoned with strong label flips and no boosting, plus severe drift. **B:** "A Little Is Enough" on clients 0, 1, 4 and 5 (z = 0.84), plus severe drift. **C:** strong poisoning plus severe drift with attacks from round 1, testing CAGA's warm-up. Pass rule, fixed in advance: CAGA beats both norm clipping and FLTrust with a 95% interval clear of zero in A or B.
- [x] **Decisive attack tests on ogbl-ddi (2026-10-14), held-out seeds 2–4.**
  - **A** (4 unboosted poisoners): overall Hits@20 ties for all methods. CAGA trusts fake edges less than FLTrust on all 3 seeds (−0.012, interval −0.017 to −0.002) and far less than norm clipping (−0.094), which fails against normal-size updates.
  - **C** (attack from round 1): CAGA beats FedAvg (+0.075) and norm clipping (+0.013, all 3 seeds) and ties FLTrust; the warm-up is not a weakness.
  - **B** ("A Little Is Enough"): CAGA 0.097 test Hits@20 against FLTrust 0.061, norm clipping 0.061 and FedAvg 0.060. **CAGA beats all three on every seed** (vs FLTrust +0.036, interval +0.010 to +0.070), and blocks the attackers in 100% of rounds. **The pre-registered pass rule is met.**
- [x] **GPU variant study (DrugBank, seeds 10–11, Kaggle):** validation AUROC 0.88–0.94. FLTrust and CAGA beat FedAvg and norm clipping under every attack; CAGA vs FLTrust is −0.004 (moderate), −0.001 (A) and +0.007 (B). The class-aware probe is exploitable: attackers inject fake edges into classes they do not hold, which it ignores. Reputation fixed the false alarms (honest clients 100% accepted) but let attackers through. **Decision: plain CAGA stays; both variants become negative ablations.**
- [x] **DrugBank CPU tuning (2026-10-14), seeds 10–11, validation AUROC.** Centralized (1 client) 0.967; clean FedAvg 0.935, so federation costs about 0.03. Severe fingerprint drift alone costs FedAvg only 0.005. Under moderate poisoning plus drift: FedAvg 0.925, norm clip ×1.0 0.930 (×2.0 0.926), FLTrust 20k 0.937 (10k 0.936), CAGA warm-up 5 0.933 (`tol_z` 1 0.933). **CAGA without warm-up collapsed on seed 11 (0.665)**, so the warm-up is necessary on DrugBank. The frozen ogbl-ddi settings carry over unchanged; the confirmation notebooks already use them.
- **Decision (2026-10-14): keep CAGA as the method.** Claim: probe-based gating catches crafted attacks (A Little Is Enough) that fool magnitude-based and similarity-based defenses, and matches them elsewhere.
- [x] **Confirmation on fresh seeds 5–7 (Kaggle GPU), criteria fixed before running:** (outcome below: **fails**)
  1. On test B, CAGA beats FLTrust and norm clipping on both datasets, with the paired interval over the 3 seeds clear of zero (DrugBank: test AUROC; ogbl-ddi: test Hits@20).
  2. On the other attacks, CAGA is no more than 0.01 AUROC (DrugBank) or 0.02 Hits@20 (ogbl-ddi) worse than FLTrust.
  3. Stress tests: with only 2 attackers (`testB2`) and with a larger shift, z = 2 (`testBz2`), CAGA stays at least as good as FLTrust.

  If 1 fails on either dataset, the claim narrows to the dataset where it holds, or the paper becomes the empirical study.

  **Outcome (2026-10-14): criterion 1 fails on both datasets, so the crafted-attack method claim is dropped.** All 144 runs complete (ogbl-ddi 60, DrugBank 84; 40 rounds each); results in the git-ignored `results/kaggle_confirm_s5-7/`. Metric: last-10-round mean of test Hits@20 (ogbl-ddi) or test AUROC (DrugBank). Differences are CAGA minus the baseline, paired over seeds 5–7, with 95% t-intervals (2 degrees of freedom).

  | Dataset, attack | CAGA | FLTrust | Norm clip | FedAvg | CAGA − FLTrust | CAGA − norm clip |
  | --- | --- | --- | --- | --- | --- | --- |
  | ogbl, test B | 0.054 | 0.026 | 0.042 | 0.051 | **+0.028** [+0.002, +0.054] | +0.013 [−0.042, +0.067] |
  | DrugBank, test B | 0.910 | 0.907 | 0.875 | 0.878 | +0.002 [−0.030, +0.035] | **+0.035** [+0.009, +0.061] |
  | ogbl, clean / strong | 0.093 / 0.077 | 0.109 / 0.082 | 0.093 / 0.065 | 0.086 / 0.041 | −0.016 / −0.005 | +0.000 / +0.012 |
  | DrugBank, clean / moderate / strong / test A | 0.930 / 0.929 / 0.929 / 0.920 | 0.932 / 0.932 / 0.931 / 0.922 | 0.928 / 0.923 / 0.922 / 0.907 | 0.927 / 0.919 / 0.747 / 0.904 | −0.002 / −0.003 / −0.002 / −0.002 | +0.002 / +0.005 / +0.007 / +0.013 |
  | ogbl, test B2 / test Bz2 | 0.063 / 0.048 | 0.040 / 0.037 | 0.074 / 0.023 | 0.081 / 0.027 | +0.023 / +0.012 | −0.011 / +0.026 |
  | DrugBank, test B2 / test Bz2 | 0.924 / 0.884 | 0.925 / 0.914 | 0.922 / 0.708 | 0.922 / 0.651 | −0.001 / **−0.029** | +0.002 / +0.177 |

  1. **Fails.** CAGA must beat both FLTrust and norm clipping on test B with the interval clear of zero, on both datasets. On ogbl-ddi it beats FLTrust (all 3 seeds, now replicated on seeds 2–4 and 5–7) but not norm clipping (seed 6 −0.013). On DrugBank it beats norm clipping but only ties FLTrust (per seed −0.002, +0.017, −0.009). It holds on neither dataset in full.
  2. **Passes.** On the other attacks CAGA is at most 0.003 AUROC (DrugBank) and 0.016 Hits@20 (ogbl-ddi, clean) below FLTrust, inside the 0.01 and 0.02 margins. FLTrust is ahead in every DrugBank row, by 0.002–0.003, consistently across seeds (DrugBank moderate: interval −0.006 to −0.0004).
  3. **Fails on DrugBank.** test Bz2: −0.029 (seed 7 −0.072); test B2: −0.001, effectively a tie but not "at least as good". On ogbl-ddi CAGA stays ahead of FLTrust in both stress tests (intervals include zero).

  What else the runs show:
  - **The ogbl-ddi test-B win shrank.** On seeds 2–4 CAGA scored 0.097 against about 0.060 for every baseline; on seeds 5–7 it scored 0.054, barely above FedAvg (+0.003). It still beats FLTrust, because FLTrust collapses under A Little Is Enough on ogbl-ddi (0.026, below FedAvg), but it no longer recovers the clean level (0.093).
  - **CAGA's false alarms are worse than measured before.** Honest client-rounds not fully accepted, after round 22: ogbl-ddi 18–26%; **DrugBank 33–47%, and 47% with no attack at all.** Under test B on DrugBank it fully accepted the attackers in 16% of rounds.
  - **FLTrust is the strongest general-purpose defense on DrugBank.** It is best or tied in every row and only loses ground to CAGA under test Bz2 on ogbl-ddi.

  **Decision (2026-10-14):** following the decision points written in advance ("on neither → empirical study"), the stand-alone CAGA method paper is dropped. Paper A continues as the active plan, because it was built not to depend on this result (see "Current verdict"). These runs become its "general attacks" section and a second, honest finding: no single defense wins everywhere, and similarity-based FLTrust fails only under crafted attacks on ogbl-ddi.
- [ ] **CAGA revamp (2026-10-13), under test on GPU.** Diagnosis from test A and the held-out runs: CAGA gates whole clients although attackers poison only part of their data; it mistakes honest non-IID specialists for attackers (16–26% of honest client-rounds blocked, even with no attack); and each round is judged alone on a noisy probe. Two opt-in fixes in `src/training/caga.py`, with tests in `tests/test_caga.py`:
  - **class-aware probe** (`class_aware`): a client is judged only on probe pairs from the drug classes it holds, weighted by its share of each;
  - **reputation** (`reputation`, `rep_beta`, `rep_threshold`): a running average of each client's leave-one-out harm; only clients with a bad record may be gated, and the warm-up rounds build that record.

  Study: `scripts/gpu_variant_study.py` (Kaggle notebook `fedddi_caga_variants.ipynb`), DrugBank, tuning seeds 10–11, under moderate poisoning, test A and test B. **Any variant picked from it must be confirmed on new held-out seeds (5–7), not 2–4**, because seeds 2–4 were seen before the redesign.
- [x] **Learning-rate decay across rounds** added (`training.round_lr_decay`, default off; pilot `--round-lr-decay`; `tests/test_round_lr.py`). Next: one Kaggle run with it on, to check that it stops the late-round decline.
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
| **LFR or Zeno also resists A Little Is Enough** | They match CAGA on test B | The method claim is no longer new: argue combination search and the step-size cap, or switch to the empirical study |
| The test-B win does not replicate | Seeds 5–7 confirmation fails | Narrow the claim to the dataset where it holds, or switch to the empirical study |
| Clean-condition gain is mistaken for defense | Reviewers note CAGA also wins with no attack | Report attacked-minus-clean gaps per method alongside raw scores |
| Adaptive attacks get through | Damage accumulates below tolerance | Report it honestly and bound the damage per round |
| Training instability masks results | Collapses in clean runs | Phase 1 stability gate |
| Compute overrun | Pilot timing | Reduced-scale ablations; prioritize the joint condition |
| Single dataset | Reviewer objection | Second dataset in Phase 3 |
