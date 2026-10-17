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

**Step 5a, early finding (2026-10-14, seed 10, 3 of 20 runs): calibrated CAGA stalls training.** With no attack, `caga_cal` gated no honest client (0% of client-rounds, against 5–95% per client for `caga`), but validation AUROC froze at 0.911 from round 22 (FedAvg 0.933, `caga` 0.938): the final step-size check chose a step of 0 in most later rounds. The check uses the same paired test as the gating, so once nobody is gated, the plain average of honest non-IID updates raises the probe loss by a tiny but "significant" amount and the step is refused. Plain CAGA only kept training by gating clients until the remaining average improved the probe. Also under test B, `caga_cal` gated the drifted clients but accepted all four A Little Is Enough attackers in every round: they send identical updates, so removing any single one barely moves the aggregate (the same masking as with the hiders). This also shows why plain CAGA caught them: its over-sensitive test cut each copy for a tiny but "significant" gain; the sensitivity that catches A Little Is Enough is the one that blocks honest specialists.

**Decisions:**
- **Final check "cap" mode** (`final_check_mode`, pilot `--caga-final-check cap`): take the full step unless it raises the mean probe loss by more than 10% (the existing divergence cap). New method `caga_cal2`; `caga_harm`, `caga_harm_solo` and the step 6 harm grid now use it. Reason: same effect-size-not-certainty logic as fix 1; without it every configuration that accepts all clients stalls, which would have wasted the 120-run harm grid.
- **Kaggle baselines notebook** uses `caga_cal2` instead of `caga_cal`, and gains a `clean` condition (108 runs; DRR needs FedAvg clean, and false alarms need clean runs).
- **The CPU study keeps running unchanged** so the pre-registered comparison is complete; its `caga_cal` rows will be read as the stalled version. The pre-registered fallback "try k = 2" is not pursued: neither observed failure (the stall, the masking) depends on k.
- **Open: coordinated identical updates.** A group-aware test (leave out clusters of near-identical updates) is the natural fix for A Little Is Enough masking; deferred until the step 6 results show whether it is needed for the paper (plain `caga` stays in every table).

**Step 5a result (2026-10-14, CPU, seeds 10–11, all 20 runs): `caga_cal` fails and is not adopted.**

| Attack | Val AUROC caga / cal | Val MDIR caga / cal | Honest false exclusion caga / cal | Attackers fully accepted caga / cal |
| --- | --- | --- | --- | --- |
| clean | 0.935 / 0.924 | 0.474 / 0.537 | 0.50 / 0.005 | – |
| moderate | 0.935 / 0.927 | 0.445 / 0.496 | 0.32 / 0.00 | 0.03 / 0.75 |
| testB | 0.915 / 0.908 | 0.560 / 0.588 | 0.49 / 0.01 | 0.10 / 1.00 |
| hideHigh4 | 0.932 / 0.925 | 0.518 / 0.535 | 0.34 / 0.00 | 0.61 / 1.00 |

(i) passes (false exclusion 0.5% with no attack); (ii) fails (attackers excluded in 0% of post-onset rounds under test B and 25% under moderate, against ≥ 80% required); (iii) fails (AUROC 0.007–0.011 below `caga` under every attack). The final step was refused (step size 0) in up to 89% of rounds after round 22, which explains the AUROC loss; the cap-only fix (`caga_cal2`) is untested and runs in the Kaggle baselines notebook. Even plain `caga` lets the four hiders through in 61% of rounds (validation MDIR 0.518 vs 0.474 clean), so neither version is a good defense against `hideHigh4`; that is the harm gate's job (step 6). **Decision: plain `caga` stays CAGA's default; `caga_cal` is a negative ablation; `caga_cal2` is judged on the Kaggle results with the same three criteria.** Ablation `caga_out` (outlier gate without the warm-up baseline): with no attack, false exclusion 1.0% and AUROC 0.931 (−0.004 vs `caga`); under test B it also accepted the attackers in 100% of rounds and stalled (step 0 in 74–100% of later rounds). So the baseline is not what lets A Little Is Enough through: the cross-client outlier test itself misses four identical updates, which a cluster-level test would address.

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

**Step 6 set-up (2026-10-14).** Two Kaggle notebooks, built by `scripts/make_kaggle_bundle.py`: `fedddi_paperA_tune_baselines.ipynb` (108 runs: FedAvg, norm clip, FLTrust, LFR, Zeno, median, `caga`, `caga_cal2` × `clean` and the five attacks × seeds 10–11) and `fedddi_paperA_tune_harm.ipynb` (120 runs: `harm_{loo,solo}_t{005,01,02}_h{3,5}`). Both need the relabelled `drugbank_ddi.pt` on Kaggle (the notebooks check). Selection rule, fixed before results, extending the one above to the harm mode: pick the (mode, harm_tol, probe_high_frac) with the lowest mean validation MDIR over `hideHigh` and `hideHigh4`, among settings whose mean validation AUROC is within 0.005 of FLTrust's under every one of the five attacks; ties within 0.005 MDIR go to the setting with the lower honest false-exclusion rate. If no setting meets the AUROC constraint, Paper A goes ahead with the threat/metric/benchmark framing (criteria 1, 2, 4) and harm-gated CAGA is reported as a negative result.

**Step 7: main results on fresh seeds 12–16 (5 seeds; never used before).**
- DrugBank random split: every attack in the scope table plus `clean`, every baseline plus `caga` and `caga_harm`.
- DrugBank cold-drug split: `clean`, `hideHigh`, `testB`.
- ogbl-ddi: the general attacks only (no risk labels): `clean`, `strong`, `testA`, `testB`, `minmax`.
- ✅ Add `signflip`, `gauss`, `minmax`, `minsum` and `ipm` attack presets first (step 4 style, in the pilot's update-replacement hook next to `lie_updates`). Done 2026-10-14: `src/training/model_attacks.py`, `tests/test_model_attacks.py`, pilot `--model-attack --model-attack-clients --ipm-eps --gauss-scale`, presets of the same names (clients 0, 1, 4, 5 attack from round 12, severe drift on 2–3, like test B). Definitions: sign flip = the attacker's own update reversed; Gaussian = noise with the honest updates' per-coordinate spread; IPM = −0.5 × the honest mean (Xie et al.); min-max / min-sum = honest mean + γ·(−std, unit length) with the largest γ keeping the attacker within the honest clients' largest pairwise distance / largest sum of squared distances (Shejwalkar & Houmansadr). All are omniscient, as A Little Is Enough.

**Step 7 notebooks (built 2026-10-14; `results/kaggle/fedddi_main_*.ipynb`, 840 runs, seeds 12–16, frozen settings).** Each notebook can resume from an earlier session's output attached as input (step 3 of the notebook); ogbl-ddi is split one attack per notebook so no session nears Kaggle's 12-hour limit. Run them in this order; the first three hold the paper's core tables.

| # | Notebook | Content | Runs | Est. GPU hours |
| --- | --- | --- | --- | --- |
| 1 | `main_01_db_core_a` | DrugBank random: clean, drift, hideHigh, hideHigh4 × all 9 methods | 180 | 6–7 |
| 2 | `main_02_db_core_b` | DrugBank random: fabHigh, moderate, testB × 9 | 135 | 5–6 |
| 3 | `main_03_db_cold` | DrugBank cold-drug: clean, hideHigh, testB × 9 | 135 | 5–6 |
| 4 | `main_04_db_model_a` | DrugBank random: sign flip, Gaussian, IPM, min-max × 6 | 120 | 4–5 |
| 5 | `main_05_db_model_b` | DrugBank random: min-sum, strong, test A, hideHighLIE × 6 | 120 | 4–5 |
| 6–10 | `main_06_ogbl_clean`, `07_ogbl_strong`, `08_ogbl_testA`, `09_ogbl_testB`, `10_ogbl_minmax` | ogbl-ddi: one attack each × 6 | 30 each | 3–4 each |

**Step 7 result, notebook 6 `main_06_ogbl_clean` (2026-10-15, Kaggle; 30/30 runs, 40 rounds each; git-ignored `results/kaggle_main/main_06_ogbl_clean/`).** Metric: test Hits@20, last-10-round mean, paired over seeds 12–16 with 95% t-intervals (4 df).

| Method | Test Hits@20 | vs FedAvg [95% CI] | vs FLTrust [95% CI] | Not fully accepted (honest) |
| --- | --- | --- | --- | --- |
| FedAvg | 0.116 | – | −0.009 | 0 |
| FLTrust | 0.125 | +0.009 [−0.020, +0.037] | – | 0.02 |
| LFR | 0.113 | −0.003 [−0.036, +0.030] | −0.012 [−0.024, +0.000] | 0.20 (by design) |
| norm clip | 0.113 | −0.003 [−0.022, +0.016] | −0.012 [−0.045, +0.022] | 0.50 (by design) |
| Zeno | 0.108 | −0.008 [−0.035, +0.019] | −0.017 [−0.034, +0.000] | 0.20 (by design) |
| CAGA | 0.096 | −0.021 [−0.066, +0.025] | −0.029 [−0.066, +0.008] | 0.21 |

- No interval excludes zero: ogbl-ddi Hits@20 is very noisy (validation Hits@20 swings 0.06–0.18 within one run; seed spread 0.06–0.19), so 5 seeds cannot separate differences below about 0.03.
- **CAGA has the largest no-attack cost**, lower than FedAvg on 4 of 5 seeds; one seed (15: −0.084, CAGA's validation fell from 0.15 to 0.07 in the last rounds while FedAvg rose) carries most of the mean. Median paired difference vs FedAvg −0.013. Same direction as seeds 5–7 (CAGA 0.016 below FLTrust, clean ogbl-ddi), larger here. Its false alarms on ogbl-ddi (21% of honest client-rounds) are lower than on DrugBank (about 50%) but the same order as LFR/Zeno's structural 20% (they drop c = 2 clients every round).
- "Not fully accepted" for LFR, Zeno and norm clip is fixed by construction (drop c = 2 of 10; clip at the median norm), so only FLTrust and CAGA's rates are informative.
- **Decision:** reported as is; it is the no-attack cost row for ogbl-ddi. CAGA's no-attack cost goes into the paper's limitations together with the DrugBank false alarms (step 5a). No change to the frozen settings: the remaining notebooks must use the same settings for the comparison to hold.

**✅ Step 7 result, all ten notebooks (2026-10-17, Kaggle; 840/840 runs complete, 40 rounds each; git-ignored `results/kaggle_main/`; tables and paired tests from `scripts/paired_main_results.py`).** Test metrics, last-10-round means; differences paired over seeds 12–16 with 95% t-intervals (4 df) and the number of seeds (of 5) in the stated direction.

*Criterion 1 (premise), FedAvg:*
- `hideHigh4` vs drift only: **MDIR +0.060 [+0.044, +0.075], 5/5 seeds; AUROC −0.001 [−0.005, +0.003]. Passes**, as on the tuning seeds (+0.068). Against clean: MDIR +0.070, AUROC −0.006.
- `hideHigh` (2 hiders) vs drift only: MDIR +0.021 [−0.002, +0.044], 4/5 seeds. Fails the 0.05 bar again; it stays the dose-response row (2 hiders +0.02, 4 hiders +0.06).

*Criterion 2 ("accuracy hides harm"), each defense under attack vs its own clean run:* **passes for every standard defense.**

| Defense | `hideHigh4` MDIR | `hideHigh4` AUROC | `hideHigh` MDIR | `hideHigh` AUROC |
| --- | --- | --- | --- | --- |
| FLTrust | +0.101 [+0.067, +0.135] | −0.007 | +0.063 [+0.030, +0.097] | −0.005 |
| LFR | +0.112 [+0.071, +0.154] | −0.008 | +0.075 [+0.036, +0.113] | −0.005 |
| Zeno | +0.094 [+0.070, +0.119] | −0.005 | +0.072 [+0.039, +0.105] | −0.004 |
| CAGA | +0.091 [+0.052, +0.130] | −0.006 | +0.052 [+0.013, +0.090] | −0.003 |
| FedAvg | +0.070 [+0.048, +0.092] | −0.006 | +0.031 [+0.002, +0.061] | −0.005 |

All MDIR rises are positive on 5/5 seeds and every AUROC change is under 0.01. **New finding, stronger than the criterion:** the defenses lose *more* safety than FedAvg, because hiding erases their clean-data advantage. With no attack, FLTrust misses 0.417 of high-risk interactions against FedAvg's 0.453. Under `hideHigh4`, every method sits at 0.52–0.55, and no defense's MDIR is distinguishable from FedAvg's (FLTrust vs FedAvg −0.005 [−0.042, +0.033]). LFR and median make hiding *worse* than FLTrust (LFR +0.032 [+0.014, +0.049], 5/5 seeds). The defenses keep AUROC (FLTrust 0.932 vs FedAvg 0.928) while losing all of their safety advantage: that is the paper's headline.

*General attacks, DrugBank random (test AUROC; full table in the script output):*
- **FLTrust is the strongest defense on DrugBank.** It is best or tied in every row and significantly ahead of every other method under every model attack. Examples (AUROC, FLTrust minus CAGA): IPM +0.013, Gaussian +0.007, sign flip +0.010, min-max +0.007, min-sum +0.008.
- CAGA is second under the model attacks: above LFR, Zeno and norm clip under IPM and sign flip, and tied with Zeno under min-max and min-sum. Under test B (A Little Is Enough), CAGA, LFR, Zeno and FLTrust tie within 0.004.
- Median and Krum collapse under test B (AUROC 0.755 and 0.699). Krum costs 0.03 AUROC even with no attack.
- **Caveat for the paper:** MDIR must be read with AUROC. Under test B, median and Krum have the *lowest* MDIR (0.45, 0.51) with very poor AUROC, because a damaged model can still rank the high-risk pairs high.
- harm reputation δ 0.2 (negative row) behaves as on the tuning seeds: −0.010 AUROC and +0.066 MDIR vs FLTrust under test B.

*ogbl-ddi (test Hits@20):* noisy, as notebook 6 showed. The only significant ranking results:
- Under `strong`, every defense beats FedAvg (+0.06 to +0.08).
- Under min-max, LFR beats FLTrust (+0.030 [+0.008, +0.052], 5/5 seeds) and FedAvg.
- Test B: CAGA has the highest mean (0.072 vs FLTrust 0.065, LFR 0.072, FedAvg 0.056; CAGA − FLTrust +0.007 [−0.019, +0.033], 4/5 seeds), not significant. The seeds 0–4 effect (+0.036) has not replicated at that size on either confirmation (seeds 5–7, 12–16).
- With no attack, CAGA costs the most (−0.021 vs FedAvg).

*Cold-drug split:* uninformative at pilot scale. Every method has AUROC 0.52–0.61 and MDIR 0.93–0.96, so the model barely generalises to unseen drugs. No differences are interpretable. **Decision:** the cold table goes to the appendix as a limitation, and cold-drug is re-run at full scale in step 10 (hidden 500, 200k observations, 100 rounds), where the model may learn it; if not, the paper says so.

*Where CAGA stands (the user's goal of finding where it outshines others):* on seeds 12–16 it wins nothing significantly.
- **Its distinctive property is attribution:** it identifies attackers in the most attack rows. It flags the attackers in 86–100% of post-onset rounds under strong, test A, test B, moderate, sign flip, min-max and min-sum on DrugBank, and in 97–100% on ogbl-ddi.
- That comes at a 25–45% honest false-exclusion rate. LFR and Zeno, given the true attacker count, reach similar identification under strong (0.99–1.0) with near-zero false alarms.
- So CAGA's candidate claims remain the count-free ones from the attacker-count study (no c needed; degrades least as attackers grow). Step 8's attacker-count replication on seeds 12–16 tests them.

**Decisions:**
1. Criteria 1 and 2 pass on fresh seeds, so Paper A goes ahead as threat + metric + benchmark.
2. The headline result is "defenses keep accuracy but lose all of their safety advantage under coordinated hiding" (criterion 2 table plus the FLTrust-vs-FedAvg MDIR comparison).
3. FLTrust is reported as the strongest general defense.
4. CAGA is one of the benchmarked defenses; its attribution rate and count-free operation are reported as properties, not wins, unless step 8 confirms them.
5. Cold-drug goes to the appendix and step 10.

"9 methods" = FedAvg, norm clip, FLTrust, LFR, Zeno, median, Krum, CAGA, harm reputation δ 0.2; "6" drops median, Krum and harm reputation (secondary attacks and ogbl-ddi, where there are no risk labels). Decisions and reasons: the plan's full matrix (every attack × every method × 5 seeds) would be about 1,200 runs; the secondary attacks and ogbl-ddi use the six methods that matter for the comparison (median and Krum already fail on DrugBank tuning; harm reputation needs risk labels). Total about 40–50 GPU hours, more than one week of Kaggle's free GPU quota (about 30 h/week), so the order above puts the core tables first. A CPU smoke run covered every new combination (cold split with harm reputation and Krum, hideHighLIE, model attacks on ogbl-ddi with LFR and Zeno).

**Step 8: robustness of the defense itself (seeds 12–16, DrugBank).**

Implemented 2026-10-14 (tests: `tests/test_adaptive_attacker.py`, `tests/test_poisoning.py`; all 241 tests pass):
- **Adaptive hider** (`--adaptive-hide`, presets `adaptive4`, `adaptive4p`). Written against the harm-reputation gate, the current harm-defense candidate. Each round each hider sends honest mean + α·(its hiding update − honest mean), with the largest α ∈ {1, 0.75, 0.5, 0.25, 0} whose estimated harm (CCH against the honest mean, at the honest mean's 95%-specificity threshold) stays under 0.9 × (1 + δ) × the median honest client's estimated harm. It knows δ and sees the honest updates (omniscient). `adaptive4` estimates harm on the hiders' pooled own high-risk pairs (kept and hidden) and random pairs; `adaptive4p` on the server's probe of the round, which is only possible against a defense **without** `--caga-secret-probe` (the pilot refuses the combination), so the "secret probe off" arm of step 8 needs harm-reputation variants without the secret probe. The plan's per-round formulation ("hides as many high-risk pairs as it can") is implemented as this blend, because it needs no extra local training.
- **Malicious majority:** `hide6` (6 hiders) and `lie6` (A Little Is Enough on 6 clients with z fixed at 0.84, test B's value; the attack's own formula has no solution for a majority).
- **On-off:** `--on-off "12-20,30-40"` (PoisonEvent `active` windows), preset `onoff4` (four hiders). It targets the harm reputation's cumulative mean, which honest rounds dilute.
1. Adaptive attacker: knows harm_tol and estimates CCH on its own data; each round it hides as many high-risk pairs as it can while keeping its estimated CCH just under the tolerance. Report cumulative MDIR over rounds, with and without `secret_probe`.
2. Malicious majority: 6 of 10 clients hide high-risk pairs.
3. On-off attacker: attacks in rounds 12–20 and 30–40 only.

**Step 8 notebooks (built 2026-10-17; `results/kaggle/fedddi_step8_*.ipynb`, 335 runs, seeds 12–16, DrugBank random, step 7 settings).** Every reference run with identical settings (clean, hideHigh, hideHigh4, testB with FedAvg, FLTrust, LFR, Zeno, CAGA, harm reputation) is reused from step 7, not re-run. New method `harm_rep_d2_open` = harm reputation δ 0.2 with a probe the clients can know (no secret re-draw), the only defense `adaptive4p` can target; it gets clean and hideHigh4 references.

| # | Notebook | Content | Runs | Est. GPU hours |
| --- | --- | --- | --- | --- |
| 1 | `step8_01_adaptive_onoff` | `adaptive4` × FedAvg, FLTrust, LFR, Zeno, CAGA, harm rep, harm rep open; `adaptive4p`, clean, hideHigh4 × harm rep open; `onoff4` × the six defenses | 80 | 4–5 (the adaptive attacker scores 5 candidates per hider per round) |
| 2 | `step8_02_majority` | `hide6`, `lie6` × the six defenses + LFR/Zeno with c = 2 and c = 4; hideHigh4 and testB × LFR/Zeno c = 2; hideHigh × LFR/Zeno c = 4 | 130 | 5–6 |
| 3 | `step8_03_few` | `hide1`, `lie1`, `testB2` × FedAvg, FLTrust, CAGA, LFR, Zeno (true c), LFR/Zeno c = 4; hide1, lie1 × LFR/Zeno c = 2 | 125 | 5 |

Decisions: the attacker-count replication joins step 8 because CAGA's three candidate claims (step 6, attacker-count study) need seeds 12–16 and share the majority runs. Median and Krum are left out (they fail on DrugBank in step 7). With m = 2, "c = 2" equals the true count, so those runs are not duplicated.

**Analysis fixed before results (2026-10-17).** All on test metrics, last-10-round means, paired over seeds 12–16 with 95% t-intervals; "beats" means the interval is clear of zero. Hiding is judged on MDIR, A Little Is Enough on AUROC.
- **Claim 1, malicious majority:** CAGA has the lowest MDIR under `hide6` (beats FLTrust, FedAvg, and LFR and Zeno at every count given: true, 2 and 4), and the highest AUROC under `lie6` (same comparisons). **Claim 1b, graceful degradation:** CAGA's MDIR rise from m = 1 to m = 6 (hide1 → hide6, per seed) is smaller than FLTrust's and FedAvg's. Claimed only for the parts that pass.
- **Claim 2, detecting few attackers:** at m = 1 and 2 (hide1, hideHigh, lie1, testB2), CAGA excludes the attackers in ≥ 50% of post-onset rounds on ≥ 4 of 5 seeds, while LFR and Zeno with the true count do so in ≤ 10%. Reported with each method's honest false-exclusion rate. The claim needs every m in at least one family.
- **Claim 3, under-estimated count:** CAGA beats LFR c = 2 and Zeno c = 2 at m = 4 and m = 6 in both families (hideHigh4, hide6, testB, lie6). If it passes only in one family, the claim names that family.
- **Criterion 4, adaptive attacker (characterisation, not pass/fail; see step 6d):** for each defense, the MDIR rise under `adaptive4` against its own clean run, the hiders' mean α per round, and the attackers' exclusion rate. The criterion as written ("with `secret_probe`, MDIR stays at least 50% below FedAvg's") is evaluated on the MDIR *rise over clean*: harm reputation's rise under `adaptive4` ≤ 0.5 × FedAvg's. Secret-probe value: `adaptive4p` vs `adaptive4` against `harm_rep_d2_open` (the attacker's best information), and `harm_rep_d2_open` vs `harm_rep_d2` under `adaptive4`.
- **On-off** (note: as in every hiding preset, hiders 0 and 4 start at round 12 and hiders 1 and 5 at round 22, the protocol's gradual attacker, so in `onoff4` hiders 1 and 5 attack only in rounds 30–40; the CPU smoke run of every new combination confirmed this and that the adaptive α responds to the budget): MDIR in the on windows (rounds 12–20, 30–40) and off window (21–29), and attackers' exclusion per window, for each defense against `hideHigh4` (always on). Descriptive.

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

**Step 6a result (2026-10-14, Kaggle T4, `results/kaggle_tune_baselines/`, 80 runs, seeds 10–11, validation, last-10-round means).** Run from the earlier zip, so it has `caga_cal` (the stalling version) instead of `caga_cal2` and no `clean` condition; both are in the new `fedddi_paperA_tune_extra.ipynb` (26 runs).

| Method | Val AUROC: fabHigh / hideHigh / hideHigh4 / moderate / testB | Val MDIR: fabHigh / hideHigh / hideHigh4 / moderate / testB | Attackers excluded, hideHigh4 / testB | Honest false exclusion (range) |
| --- | --- | --- | --- | --- |
| FedAvg | 0.932 / 0.930 / 0.927 / 0.923 / 0.884 | 0.430 / 0.506 / 0.536 / 0.504 / 0.676 | 0 / 0 | 0 |
| norm clip | 0.932 / 0.930 / 0.927 / 0.929 / 0.885 | 0.445 / 0.519 / 0.551 / 0.484 / 0.677 | 0.33 / 0 | 0.20–0.85 |
| FLTrust | **0.938 / 0.938** / 0.934 / **0.938** / 0.907 | **0.410 / 0.443** / 0.499 / **0.424** / 0.579 | 0.08 / 0.05 | 0–0.01 |
| LFR | 0.937 / 0.934 / 0.930 / 0.934 / **0.916** | 0.412 / 0.500 / 0.533 / 0.458 / **0.536** | 0.26 / 0.68 | 0–0.26 |
| Zeno | 0.936 / 0.933 / **0.936** / 0.933 / 0.914 | 0.414 / 0.502 / **0.495** / 0.452 / 0.545 | 0.26 / 0.81 | 0–0.26 |
| median | 0.934 / 0.930 / 0.929 / 0.929 / 0.838 | 0.450 / 0.527 / 0.555 / 0.506 / 0.619 | – | 0 |
| CAGA | 0.935 / 0.932 / 0.931 / 0.934 / 0.914 | 0.429 / 0.482 / 0.506 / 0.448 / 0.543 | 0.36 / 0.83 | 0.29–0.34 |
| `caga_cal` (stalls) | 0.932 / 0.928 / 0.922 / 0.927 / 0.902 | 0.442 / 0.509 / 0.560 / 0.491 / 0.605 | 0 / 0 | 0–0.03 |

What it shows:
- **The novelty risk materialised: LFR and Zeno match CAGA under A Little Is Enough** (CAGA − LFR −0.001 and −0.003, CAGA − Zeno −0.000 and −0.000 on the two seeds), and they exclude the attackers about as often (68–81% vs 83%). With the seeds 5–7 failure, this settles it: CAGA's crafted-attack advantage is not a contribution of Paper A. Effect-based defenses as a family resist A Little Is Enough; FLTrust (similarity-based) does not on ogbl-ddi and is 0.007–0.009 behind here.
- **FLTrust is the strongest general defense on DrugBank**: best AUROC under four attacks, best MDIR under fabHigh, hideHigh and moderate, and almost no false alarms.
- **No defense undoes `hideHigh4`.** Best MDIR 0.495 (Zeno) and 0.499 (FLTrust), against about 0.41–0.43 for the same defenses under fabHigh; attackers are excluded in at most 36% of rounds. This is the gap the harm gate has to close, and it supports criterion 2 ("accuracy hides harm"): FLTrust and Zeno keep AUROC at 0.934–0.936 while their MDIR is 0.08–0.09 higher than under fabHigh. **Provisional**: fabHigh is not the clean reference; the clean runs in step 6c decide it.
- Coordinate median collapses under test B (0.838) and is the worst on MDIR; norm clipping wrongly clips 85% of honest client-rounds under test B.
- For the step 6 selection rule, harm-gated CAGA must keep validation AUROC ≥ FLTrust − 0.005 under every attack: ≥ 0.933 / 0.933 / 0.929 / 0.933 / 0.902. Plain CAGA misses it by 0.001 under hideHigh.

**Step 6c result (2026-10-14, Kaggle, 26 runs: clean for every method, `caga_cal2` under the five attacks; merged into `results/kaggle_tune_baselines/`).**
- **Criterion 2 ("accuracy hides harm") holds on the tuning seeds** (provisional; the formal test is on seeds 12–16 in step 7). Against each defense's own clean run (validation): FLTrust under `hideHigh4` AUROC −0.002, MDIR +0.044 (seeds: +0.021, +0.067); LFR under `hideHigh4` −0.007 / +0.074 and under `hideHigh` −0.002 / +0.040; Zeno under `hideHigh` −0.002 / +0.034. CAGA, norm clipping and median also meet it under `hideHigh4`.
- **Against clean, plain FedAvg barely moves under `hideHigh`** (MDIR −0.001; `hideHigh4` +0.030, AUROC −0.006). The defenses that use clean server data are better when clean (FLTrust clean MDIR 0.455 vs FedAvg 0.507), so they have more to lose. Fabrication *lowers* MDIR (fake high-risk pairs raise scores in high-risk classes), so fabHigh is not a usable reference (as suspected in step 6a).
- **No-attack costs (val AUROC vs FedAvg 0.933):** FLTrust +0.004, LFR +0.003, Zeno +0.002, median +0.002, CAGA +0.002, norm clip 0.000. False exclusion with no attack: CAGA 49%, norm clip 50%, LFR and Zeno 20% (they drop c = 2 clients every round by design), FLTrust 4%.
- **`caga_cal2` fails the step 5a criteria:** (i) passes (0% false exclusion); (ii) fails (attackers excluded in 0.7% of rounds under test B, 8% under moderate); (iii) fails (AUROC −0.007 moderate, −0.016 test B vs `caga`). The stall is gone (clean AUROC 0.933, like FedAvg), but it is effectively FedAvg. **Decision: the calibrated line of CAGA is closed** (`caga_cal`, `caga_out`, `caga_cal2` are negative ablations). CAGA's detection and its false alarms come from the same sensitivity; cross-client calibration removes both. Plain `caga` stays the default.

**Goal added (2026-10-14, user): find where CAGA outshines the other defenses**, alongside making the paper. Evidence so far: CAGA excludes the attacking clients most consistently across attack types (hideHigh 0.58, hideHigh4 0.36, moderate 1.00, testB 0.83; the next best per column is 0.13–0.88 and never the same defense), at the cost of 29–49% false exclusion; it is not better on AUROC or MDIR on DrugBank, and LFR/Zeno match it under A Little Is Enough.

**Attacker-count study (notebook `fedddi_paperA_attacker_count.ipynb`, 102 runs, seeds 10–11; presets `hide1`, `hide6`, `lie1`, methods `lfr_c2`, `lfr_c4`, `zeno_c2`, `zeno_c4`, `krum_c2`; pilot `--poisoned-count`).** Hypothesis: LFR, Zeno and Krum need the number of attackers c; in our runs so far they got the true number (an oracle). With a fixed guess they should degrade when the guess is wrong (too small: extra attackers get through; too large: honest clients are dropped), while CAGA, which needs no count, should not. Analysis fixed before results: for each attack family (hiding: m = 1, 2, 4, 6; A Little Is Enough: m = 1, 2, 4), compare `caga` with the guessed-c versions (c = 2, and c = 4 where m ≤ 2) on validation MDIR (hiding) or validation AUROC (A Little Is Enough), attackers excluded and honest false exclusion; report the oracle versions alongside. **CAGA "outshines" a defense if it beats that defense's guessed-c version on the primary metric for every m in a family, by more than the seed-to-seed spread of the difference, and the paper may claim it only after it replicates on seeds 12–16 in step 7.** m = 6 is a malicious majority, outside every method's design; it is reported, not used for the claim. Runs that already exist with identical settings (hideHigh, hideHigh4, testB with the oracle versions, FedAvg, FLTrust, CAGA) are reused from step 6a.

**Step 6b result (2026-10-14, Kaggle, 120 runs, `results/kaggle_tune_harm_oldcheck/`): not usable for selection.** The notebook ran from a zip older than the cap-only final check (no `caga_final_check` in settings), so the final step was refused in 9–82% of rounds after round 22 and AUROC/MDIR are confounded (val AUROC 0.896–0.930, below FLTrust − 0.005 everywhere). Detection, which happens before the final check, is still informative:
- **Leave-one-out harm does not separate hiders from honest clients**: per-round AUC (attacker vs honest client CCH) 0.52 under `hideHigh`, 0.53 under `hideHigh4`, 0.26 under test B. Its apparent catch rate (up to 57% under `hideHigh4` at tolerance 0.005) came with as many honest exclusions (45%): chance. **The `harm_loo` variants are dropped.**
- **Solo harm carries signal**: per-round AUC 0.77 (`hideHigh`), 0.67 (`hideHigh4`), 0.99 (moderate), 0.94 (test B). Averaged per client over rounds 22–40 (offline, these runs): AUC 0.92 (`hideHigh`), 0.81 (`hideHigh4`), 0.88 (fabHigh), 1.00 (moderate), 0.75 (test B).
- **But the `harm_solo` gate caught nobody (0%)**, because of the step 5a calibration: the severely drifted honest clients have 3–4× the hiders' harm (solo CCH about 275–326 vs 80 per round), so they set the cross-client scale and the hiders never reach z > 3. Subtracting each client's warm-up harm makes it worse (offline AUC 0.34–0.40, below chance), so the warm-up baseline is dropped for harm.
- **Finding for the paper:** under severe feature drift, honest drifted clients cause far more clinical harm (high-risk interactions hidden) than deliberate hiders. A harm gate that acts on clinical harm will, and arguably should, down-weight them; the paper must report drifted and attacking clients separately.
- **Finding for the paper:** leave-one-out effect scores, which LFR, Zeno-style and CAGA rely on, are masked when several attackers act alike; per-client (solo) scores are not.

**Built (2026-10-14) as step 6d; notebook `fedddi_paperA_harm_rep.ipynb`, 36 runs (`harm_rep_d1`, `d2`, `d3` × clean and the five attacks × seeds 10–11).** `CAGAHyperparams.harm_rep`, `harm_rep_delta`; pilot `--caga-harm-rep --caga-harm-rep-delta`; tests in `tests/test_caga.py`. The running mean starts after warm-up and is a plain cumulative mean (an on-off attacker could dilute it with honest rounds; step 8 tests that). A flagged client is still only shrunk or dropped when removing it lowers the risk-weighted probe loss, as for every CAGA move. `caga_cch` reports the running mean in this mode. **Selection, fixed before results:** the step 6 rule unchanged (lowest mean validation MDIR over `hideHigh` and `hideHigh4`, among the δ values with validation AUROC within 0.005 of FLTrust under all five attacks; ties within 0.005 MDIR go to the lower honest false exclusion; FLTrust's values from step 6a). If no δ qualifies, the harm defense is reported as a negative result and Paper A takes the threat/metric/benchmark framing.

The proposal as first written: "harm reputation": each client's running mean of solo CCH from the end of warm-up, no baseline and no cross-client z-score; a client is gated when its running mean exceeds (1 + δ) × the median client's, δ ∈ {0.1, 0.2, 0.3}; cap-only final check. About 36 Kaggle runs (3 δ × 5 attacks + clean, seeds 10–11). The step 6 selection rule above applies unchanged.

**Attacker-count study result (2026-10-14, Kaggle, 102 runs plus the step 6a runs at m = 2 and 4; `results/kaggle_attacker_count/`, combined table `combined_runs.csv`; seeds 10–11, validation).**

Hiding (validation MDIR, lower is better), by number of attackers m:

| m | FedAvg | FLTrust | CAGA | LFR c=2 / c=4 / true | Zeno c=2 / c=4 / true | Krum c=2 / true |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.490 | **0.436** | 0.469 | 0.478 / 0.478 / 0.481 | 0.491 / **0.430** / 0.469 | 0.625 / 0.606 |
| 2 | 0.506 | **0.443** | 0.482 | 0.500 / 0.484 / 0.500 | 0.502 / 0.459 / 0.502 | 0.644 / 0.644 |
| 4 | 0.536 | 0.499 | 0.506 | 0.547 / 0.533 / 0.533 | 0.534 / **0.495** / 0.495 | 0.633 / 0.614 |
| 6 | 0.580 | 0.564 | **0.537** | 0.583 / – / 0.578 | 0.578 / – / 0.542 | 0.695 / 0.711 |

A Little Is Enough (validation AUROC): m = 1: CAGA 0.931, FLTrust 0.935, LFR/Zeno 0.934–0.936 whatever the guess; m = 2: CAGA 0.930, the others 0.932–0.935; m = 4: CAGA 0.914, LFR c=2 0.906, Zeno c=2 0.900, LFR/Zeno c=4 0.916/0.914, FLTrust 0.907, Krum 0.62–0.64.

**Against the pre-registered rule** (CAGA beats the guessed-c version at every m ∈ {1, 2, 4} of a family, on both seeds):
- **Passes vs Zeno with c = 2 under hiding** (CAGA lower MDIR on both seeds at m = 1, 2, 4, and at m = 6).
- **Passes vs Krum** (both guesses, both families), but Krum fails on DrugBank even with the true count (MDIR 0.61–0.71; AUROC 0.62 under test B), so this says nothing about the count.
- **Fails vs LFR with c = 2 under hiding** only at m = 1 on seed 11 (−0.004); it passes at m = 2, 4 and 6.
- **Fails vs any c = 4 guess, and vs c = 2 under A Little Is Enough at m = 1–2**, where LFR and Zeno are 0.002–0.006 AUROC ahead.

What it means:
- **Under-estimating the count is what hurts LFR and Zeno**: at m = 4 with c = 2 they lose 0.04–0.05 MDIR under hiding (0.534–0.547 vs CAGA 0.506) and 0.008–0.015 AUROC under A Little Is Enough. A defender who guesses high (c = 4) does as well as CAGA or better, with fewer false alarms (15–29% vs CAGA's 29–41%). So "needs no count" is an advantage only against an optimistic guess. Over-guessing even helps MDIR under hiding (Zeno c = 4 is the best at m = 1 and 4), because dropping extra clients also drops the drifted ones, which cause the most harm.
- **CAGA degrades least as attackers increase.** From m = 1 to 6 its MDIR rises by 0.068, against 0.128 for FLTrust and 0.090 for FedAvg; it goes from mid-table at m = 1 to the lowest MDIR of all methods at m = 6 (0.537; FLTrust 0.564, Zeno with the true count 0.542, LFR 0.578). At m = 6 it beats FLTrust and LFR on both seeds (by 0.015–0.051) and Zeno-with-true-count on one (seed 10 +0.003 worse, seed 11 0.013 better). m = 6 is a malicious majority and was declared "report, not claim"; it is tested formally in step 8 (malicious majority) on seeds 12–16.
- **Detection when attackers are few**: with one or two attackers, LFR and Zeno given the true count almost never remove the attacker (0–4% of attacker rounds; leave-one-out picks other clients, likely the drifted ones), while CAGA removes them in 58–100% (hide1 0.71, hideHigh 0.58, lie1 1.00, testB2 0.95). Its accuracy does not benefit, because one or two attackers barely hurt the model.

**Decision (2026-10-14):** candidate CAGA claims for step 7–8, to be tested on seeds 12–16 and stated only if they replicate: (1) "CAGA degrades least as the number of attackers grows, and has the lowest MDIR with a malicious majority" (add `hide6` and a 6-attacker A Little Is Enough variant with a fixed z to step 8); (2) "CAGA identifies attackers even when they are few, where leave-one-out defenses do not"; (3) "against an under-estimated attacker count, CAGA beats LFR and Zeno". The claim "CAGA needs no attacker count" is not made without the qualifier in (3), because a high guess matches it.

**Step 6d result (2026-10-14, Kaggle, 36 runs, `results/kaggle_harm_rep/`, correct code: cap-only final check, harm reputation on): no δ qualifies; the harm defense is a negative result.**

| Method | Val AUROC − FLTrust: fabHigh / hideHigh / hideHigh4 / moderate / testB | Val MDIR: clean / hideHigh / hideHigh4 / testB | Attackers excluded: hideHigh / hideHigh4 / moderate / testB | Honest false exclusion (clean) |
| --- | --- | --- | --- | --- |
| harm rep δ 0.1 | −0.004 / −0.005 / −0.005 / −0.005 / **−0.014** | 0.496 / 0.470 / 0.510 / 0.635 | 0.36 / 0.22 / 0.93 / 0.00 | 0.15 |
| harm rep δ 0.2 | −0.004 / −0.006 / −0.005 / −0.004 / **−0.010** | 0.478 / 0.471 / 0.519 / 0.587 | 0.46 / 0.13 / 0.91 / 0.00 | 0.09 |
| harm rep δ 0.3 | −0.005 / −0.006 / −0.005 / −0.005 / **−0.011** | 0.479 / 0.482 / 0.526 / 0.619 | 0.17 / 0.09 / 0.58 / 0.00 | 0.05 |
| CAGA | −0.004 / −0.006 / −0.003 / −0.004 / +0.007 | 0.462 / 0.482 / 0.506 / 0.543 | 0.58 / 0.36 / 1.00 / 0.83 | 0.49 |
| FLTrust | 0 | 0.455 / **0.443** / 0.499 / 0.579 | 0.13 / 0.08 / 0.46 / 0.05 | 0.04 |
| Zeno | −0.003 / −0.006 / +0.002 / −0.005 / +0.008 | 0.468 / 0.502 / **0.495** / 0.545 | 0.04 / 0.26 / 0.50 / 0.81 | 0.20 |

- **Selection rule:** every δ misses the AUROC constraint under test B (0.010–0.014 below FLTrust) and narrowly under hideHigh (0.005–0.006). (Plain CAGA would also miss it under hideHigh, by 0.0009.)
- **Even ignoring the constraint, harm reputation does not beat FLTrust on hiding MDIR:** best mean over hideHigh and hideHigh4 is 0.490 (δ 0.1) against FLTrust 0.471, Zeno 0.499, CAGA 0.494. Per seed, δ 0.1 beats FLTrust only under hideHigh4 on seed 11 (0.476 vs 0.485).
- **It does not see A Little Is Enough** (0% of attacker rounds excluded, AUROC 0.893–0.897), because a crafted update need not hide high-risk pairs; a harm-only gate is blind to attacks that are not about harm.
- **What worked:** far fewer false alarms than CAGA (4–15% vs 29–49% of honest client-rounds), and it still catches moderate poisoning (58–93%). Its no-attack cost is small (val AUROC 0.933–0.935 vs FedAvg 0.933).
- **Representative δ for later tables: 0.2**, by the selection rule's own tie-break with the AUROC constraint dropped (δ 0.1 and 0.2 are within 0.005 MDIR, 0.490 vs 0.495; δ 0.2 has the lower honest false exclusion, 8% vs 11%).

**Decision (2026-10-14), as pre-registered:** "If no δ qualifies, the harm defense is reported as a negative result and Paper A takes the threat/metric/benchmark framing." **Pass criterion 3 cannot be met. Paper A goes ahead as threat + metric + benchmark (criteria 1, 2 and 4)**, with these defense findings as results: (a) no tested defense undoes coordinated hiding (best MDIR about 0.50 vs 0.45 clean, FLTrust); (b) leave-one-out scores are masked by coordinated attackers; (c) harm-aware gating trades false alarms for blindness to non-harm attacks; (d) CAGA's candidate claims from the attacker-count study (graceful degradation, detection of few attackers). Criterion 4 (adaptive attacker) is re-scoped: there is no successful harm defense to attack, so step 8 tests the adaptive hider against harm reputation δ 0.2 and CAGA as a robustness *characterisation*, not a pass/fail.

**Frozen for step 7 (fresh seeds 12–16):** methods FedAvg, norm clip ×1.0, FLTrust (root 20k), LFR, Zeno (c = true count, plus `_c2` variants where the count differs from 2), coordinate median, Krum, CAGA (warm-up 5, whole-client gating, `tol_z` 2, final check "z"), harm reputation δ 0.2 (negative-result row). Learning rate 0.003, 2 local epochs, 40 rounds, attack onset round 12, root fraction 0.12, probe 4,000.

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
