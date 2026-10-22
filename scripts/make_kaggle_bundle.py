"""Build the Kaggle bundle: results/kaggle/fedddi_code.zip (tracked and untracked-but-not-ignored files,
   Linux line endings, never data/) plus the Kaggle notebooks. Run from the repository root:
    python scripts/make_kaggle_bundle.py
"""
import json, subprocess, zipfile

NL = "\n"
import os
os.makedirs('results/kaggle', exist_ok=True)
files = sorted(set(subprocess.run(['git', 'ls-files'], capture_output=True, text=True).stdout.split())
               | set(subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], capture_output=True, text=True).stdout.split()))
with zipfile.ZipFile('results/kaggle/fedddi_code.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        data = open(f, 'rb').read()
        if f.endswith(('.sh', '.py', '.yaml', '.txt', '.md', '.toml')):
            data = data.replace(b'\r\n', b'\n')
        z.writestr('fedddi/' + f, data)
print(len(files), 'files zipped')


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}
def bash(*lines): return code(NL.join(["%%bash", "set -e"] + list(lines)))


COPY = bash("SCRIPT=$(find /kaggle/input -path '*fedddi/scripts/kaggle_gpu_checks.sh' | head -1)",
            "[ -n \"$SCRIPT\" ] || { echo 'fedddi_code.zip dataset not found: use Add Input'; exit 1; }",
            "SRC=$(dirname \"$(dirname \"$SCRIPT\")\")", "echo \"code found at $SRC\"",
            "rm -rf /kaggle/working/fedddi && cp -r \"$SRC\" /kaggle/working/fedddi",
            "command -v nvidia-smi >/dev/null || { echo 'No GPU: set Settings > Accelerator to GPU T4 and run again'; exit 1; }",
            "nvidia-smi --query-gpu=name,memory.total --format=csv")
COPY_DB = bash("DB=$(find /kaggle/input -name drugbank_ddi.pt | head -1)",
               "[ -n \"$DB\" ] || { echo 'drugbank_ddi.pt not found: attach your private DrugBank dataset'; exit 1; }",
               "mkdir -p /kaggle/working/fedddi/data/drugbank && cp \"$DB\" /kaggle/working/fedddi/data/drugbank/",
               "echo \"copied $DB\"")
COPY_DD = bash("DD=$(find /kaggle/input -name ddinter_ddi.pt | head -1)",
               "[ -n \"$DD\" ] || { echo 'ddinter_ddi.pt not found: add it to your private DrugBank dataset (it holds DrugBank-derived features)'; exit 1; }",
               "mkdir -p /kaggle/working/fedddi/data/ddinter_graph && cp \"$DD\" /kaggle/working/fedddi/data/ddinter_graph/",
               "echo \"copied $DD\"")
CHECK_RISK = code(NL.join([
    "import torch",
    "b = torch.load('/kaggle/working/fedddi/data/drugbank/drugbank_ddi.pt', weights_only=False)",
    "assert 'edge_risk' in b, 'drugbank_ddi.pt has no risk labels: upload the file rebuilt with prepare_drugbank.py --risk-only'",
    "print('risk labels present; high-risk share', float((b['edge_risk'] == 2).float().mean()))"]))
def RUN(methods, attacks):
    return code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 "
                f"--methods {methods} --attacks {attacks} --device cuda --parallel 3 --out /kaggle/working/results/paperA_count")


PIP = 'pip install -q torch-geometric ogb "flwr[simulation]>=1.10" hydra-core omegaconf "scikit-learn>=1.3"'
PACK = code(NL.join(["import shutil", "shutil.make_archive('/kaggle/working/results', 'zip', '/kaggle/working/results')",
                     "print('Download results.zip from the Output tab')"]))
# "!" streams output live in Kaggle's log; a %%bash cell shows it only when the whole cell ends.
def stage(name): return code(f"!cd /kaggle/working/fedddi && bash scripts/kaggle_gpu_checks.sh {name}")
HEAD = ("**Before running:** set *Accelerator* to **GPU T4** and turn *Internet* **on**; attach the `fedddi_code.zip` dataset "
        "(latest version) with **Add Input**. Run with **Save Version → Save & Run All (Commit)**. Each cell stops the run if it "
        "fails. When it finishes, download `results.zip` from the **Output** tab.")
PRIVATE = ("Also attach your **private** DrugBank dataset holding `drugbank_ddi.pt`. DrugBank data is licensed: keep that "
           "dataset and this notebook private.")

nbs = {
    'fedddi_lrdecay.ipynb': [
        md("# FedDDI-CAGA: learning-rate decay check" + NL + NL + HEAD), md("## 1. Copy the code"), COPY,
        md("## 2. Install and download ogbl-ddi (~5 min)"), stage('setup'), md("## 3. Smoke test (~15 min)"), stage('smoke'),
        md("## 4. Clean FedAvg, 100 rounds, 5 local epochs, learning-rate decay across rounds (~2.5 h)"), stage('lrdecay'),
        md("## 5. Package results"), PACK],
    'fedddi_centralized.ipynb': [
        md("# FedDDI-CAGA: centralized baselines on ogbl-ddi" + NL + NL + HEAD), md("## 1. Copy the code"), COPY,
        md("## 2. Install and download ogbl-ddi (~5 min)"), stage('setup'),
        bash("cd /kaggle/working/fedddi", "grep -q build_dense_edge_counts src/models/encoder.py || "
             "{ echo 'old fedddi_code.zip: upload the latest one'; exit 1; }"),
        md("## 3. Centralized at pilot scale: the step 7 model and features, one client holding all ten clients' "
           "observations, seeds 12 to 16 (5 runs, about 15 min). Compare with step 7's clean FedAvg: the cost of "
           "federation."),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset ogbl-ddi --split random "
             "--seeds 12 13 14 15 16 --methods central --attacks clean --device cuda --parallel 3 "
             "--out /kaggle/working/results/cen_pilot && python scripts/summarize_variants.py /kaggle/working/results/cen_pilot"),
        md("## 4. Centralized, original model (width 500), fixed structural features, 3 runs: the cost of the "
           "pilot's smaller model and shorter training"), stage('centralized'),
        md("## 5. Centralized, original model, learned embeddings, 1 run: the cost of fixed features "
           "(compare with the OGB leaderboard)"), stage('centralized_emb'),
        md("## 6. Package results"), PACK],
    'fedddi_drugbank_smoke.ipynb': [
        md("# FedDDI-CAGA: DrugBank GPU smoke test" + NL + NL + HEAD + NL + NL + PRIVATE),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB, md("## 2. Install packages"), bash(PIP),
        md("## 3. Two-round CAGA run on DrugBank (~10 min)"), stage('drugbank_smoke'), md("## 4. Package results"), PACK],
    'fedddi_caga_variants.ipynb': [
        md("# FedDDI-CAGA: CAGA variant study on DrugBank (GPU)" + NL + NL + HEAD + NL + NL + PRIVATE + NL + NL +
           "Runs 48 short federated runs, 3 at a time on the GPU: FedAvg, norm clipping, FLTrust and five CAGA variants "
           "(plain, class-aware probe, reputation at two thresholds, both) under three attacks, on tuning seeds 10 and 11. "
           "If the session stops, run it again: finished runs are skipped."),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB,
        md("## 2. Install packages and run the fast tests"),
        bash(PIP, "cd /kaggle/working/fedddi",
             "for t in tests/test_caga.py tests/test_drugbank.py tests/test_round_lr.py; do python -m pytest \"$t\" -q -p no:cacheprovider | tail -1; done"),
        md("## 3. Variant study, 48 runs (about 2 to 3 hours)"),
        bash("cd /kaggle/working/fedddi",
             "python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 --device cuda --parallel 3 --out /kaggle/working/results/gpu_variants"),
        md("## 4. Summary table"),
        code(NL.join(["%%bash", "cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/gpu_variants"])),
        md("## 5. Package results"), PACK],
    'fedddi_confirm_drugbank.ipynb': [
        md("# FedDDI-CAGA: confirmation on fresh seeds, DrugBank (GPU)" + NL + NL + HEAD + NL + NL + PRIVATE + NL + NL +
           "84 runs on seeds 5, 6 and 7 (never used before): FedAvg, norm clipping, FLTrust and CAGA under no attack, "
           "moderate and strong poisoning, test A, and A Little Is Enough in three forms. About 3 to 4 hours; progress "
           "prints as each run finishes. If the session stops, run it again: finished runs are skipped."),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB,
        md("## 2. Install packages"), bash(PIP),
        md("## 3. Confirmation runs (live progress below)"),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset drugbank --seeds 5 6 7 "
             "--methods fedavg normclip fltrust caga --attacks clean moderate strong testA testB testB2 testBz2 "
             "--device cuda --parallel 3 --out /kaggle/working/results/confirm_drugbank"),
        md("## 4. Summary table"),
        code("!cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/confirm_drugbank"),
        md("## 5. Package results"), PACK],
    'fedddi_confirm_ogbl.ipynb': [
        md("# FedDDI-CAGA: confirmation on fresh seeds, ogbl-ddi (GPU)" + NL + NL + HEAD + NL + NL +
           "60 runs on seeds 5, 6 and 7 (never used before): FedAvg, norm clipping, FLTrust and CAGA under no attack, "
           "strong poisoning, and A Little Is Enough in three forms. About 6 to 8 hours; progress prints as each run "
           "finishes. If the 12-hour session limit stops it, run it again: finished runs are skipped."),
        md("## 1. Copy the code"), COPY,
        md("## 2. Install and download ogbl-ddi (~5 min)"), stage('setup'),
        md("## 3. Confirmation runs (live progress below; features are built once per seed first)"),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset ogbl-ddi --seeds 5 6 7 "
             "--methods fedavg normclip fltrust caga --attacks clean strong testB testB2 testBz2 "
             "--device cuda --parallel 3 --out /kaggle/working/results/confirm_ogbl"),
        md("## 4. Summary table"),
        code("!cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/confirm_ogbl"),
        md("## 5. Package results"), PACK],
    'fedddi_paperA_tune_baselines.ipynb': [
        md("# FedDDI-CAGA Paper A, step 6a: baselines under safety-critical attacks (DrugBank, GPU)" + NL + NL + HEAD + NL + NL
           + PRIVATE + " **It must be the relabelled `drugbank_ddi.pt` with risk labels** (built with "
           "`scripts/prepare_drugbank.py --risk-only`); the notebook stops if the labels are missing." + NL + NL +
           "108 runs on tuning seeds 10 and 11: FedAvg, norm clipping, FLTrust, LFR, Zeno, coordinate median, CAGA and "
           "calibrated CAGA with no attack and under hideHigh, hideHigh4, fabHigh, moderate and test B. About 4 to 5 hours. If the session "
           "stops, run it again: finished runs are skipped."),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB, CHECK_RISK,
        md("## 2. Install packages"), bash(PIP),
        md("## 3. Runs (live progress below)"),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 "
             "--methods fedavg normclip fltrust lfr zeno median caga caga_cal2 "
             "--attacks clean hideHigh hideHigh4 fabHigh moderate testB "
             "--device cuda --parallel 3 --out /kaggle/working/results/paperA_tune"),
        md("## 4. Summary table"),
        code("!cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/paperA_tune"),
        md("## 5. Package results"), PACK],
    'fedddi_paperA_tune_extra.ipynb': [
        md("# FedDDI-CAGA Paper A, step 6c: clean runs and calibrated CAGA (DrugBank, GPU)" + NL + NL + HEAD + NL + NL
           + PRIVATE + " **It must be the relabelled `drugbank_ddi.pt` with risk labels.**" + NL + NL +
           "26 runs on tuning seeds 10 and 11, completing the step 6a baselines: every method with no attack (the reference "
           "for MDIR rises, DRR and false alarms), and calibrated CAGA with the cap-only final check (`caga_cal2`) under "
           "the five attacks. About 1 to 1.5 hours. Finished runs are skipped if you run it again."),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB, CHECK_RISK,
        md("## 2. Install packages"), bash(PIP),
        md("## 3. Clean runs, every method"),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 "
             "--methods fedavg normclip fltrust lfr zeno median caga caga_cal2 --attacks clean "
             "--device cuda --parallel 3 --out /kaggle/working/results/paperA_tune"),
        md("## 4. Calibrated CAGA under the five attacks"),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 "
             "--methods caga_cal2 --attacks hideHigh hideHigh4 fabHigh moderate testB "
             "--device cuda --parallel 3 --out /kaggle/working/results/paperA_tune"),
        md("## 5. Summary table"),
        code("!cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/paperA_tune"),
        md("## 6. Package results"), PACK],
    'fedddi_paperA_attacker_count.ipynb': [
        md("# FedDDI-CAGA Paper A: unknown number of attackers (DrugBank, GPU)" + NL + NL + HEAD + NL + NL + PRIVATE
           + " **It must be the relabelled `drugbank_ddi.pt` with risk labels.**" + NL + NL +
           "102 runs on tuning seeds 10 and 11. LFR, Zeno and Krum must be told the number of attackers; here they get "
           "a fixed guess (2, or 4) instead of the true number, while the true number of attackers varies (1, 2, 4, 6 for "
           "hiding high-risk interactions; 1, 2, 4 for A Little Is Enough). FedAvg, FLTrust and CAGA need no count. "
           "About 4 hours; finished runs are skipped if you run it again."),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB, CHECK_RISK,
        md("## 2. Install packages"), bash(PIP),
        md("## 3. One and six hiders, one A Little Is Enough attacker: every method"),
        RUN("fedavg fltrust caga lfr zeno krum lfr_c2 zeno_c2 krum_c2", "hide1 hide6 lie1"),
        md("## 4. Overestimated count (4) with one attacker"), RUN("lfr_c4 zeno_c4", "hide1 lie1"),
        md("## 5. Two A Little Is Enough attackers"), RUN("fedavg fltrust caga lfr zeno krum lfr_c4 zeno_c4", "testB2"),
        md("## 6. Two hiders: Krum and the overestimate"), RUN("krum lfr_c4 zeno_c4", "hideHigh"),
        md("## 7. Four attackers with the guess of 2, and Krum"), RUN("krum lfr_c2 zeno_c2 krum_c2", "hideHigh4 testB"),
        md("## 8. Krum with no attack"), RUN("krum", "clean"),
        md("## 9. Summary table"),
        code("!cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/paperA_count"),
        md("## 10. Package results"), PACK],
    'fedddi_paperA_harm_rep.ipynb': [
        md("# FedDDI-CAGA Paper A, step 6d: harm reputation (DrugBank, GPU)" + NL + NL + HEAD + NL + NL + PRIVATE
           + " **It must be the relabelled `drugbank_ddi.pt` with risk labels.**" + NL + NL +
           "36 runs on tuning seeds 10 and 11: harm-gated CAGA that gates a hospital only when its running mean "
           "clinical harm exceeds the median hospital's by more than delta (0.1, 0.2, 0.3), with no attack and under "
           "hideHigh, hideHigh4, fabHigh, moderate and test B. About 1.5 to 2 hours. Finished runs are skipped if you "
           "run it again."),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB, CHECK_RISK,
        md("## 2. Install packages and check the code version"),
        bash(PIP, "cd /kaggle/working/fedddi", "grep -q harm_rep_delta src/training/caga.py || "
             "{ echo 'old fedddi_code.zip: upload the latest one'; exit 1; }"),
        md("## 3. Runs (live progress below)"),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 "
             "--methods harm_rep_d1 harm_rep_d2 harm_rep_d3 --attacks clean hideHigh hideHigh4 fabHigh moderate testB "
             "--device cuda --parallel 3 --out /kaggle/working/results/paperA_harm_rep"),
        md("## 4. Summary table"),
        code("!cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/paperA_harm_rep"),
        md("## 5. Package results"), PACK],
    'fedddi_paperA_tune_harm.ipynb': [
        md("# FedDDI-CAGA Paper A, step 6b: harm-gated CAGA grid (DrugBank, GPU)" + NL + NL + HEAD + NL + NL + PRIVATE
           + " **It must be the relabelled `drugbank_ddi.pt` with risk labels.**" + NL + NL +
           "120 runs on tuning seeds 10 and 11: harm-gated CAGA in two modes (leave-one-out and calibrated solo) x "
           "harm tolerance 0.005 / 0.01 / 0.02 x high-risk probe share 0.3 / 0.5, under hideHigh, hideHigh4, fabHigh, "
           "moderate and test B. About 6 to 8 hours; if the 12-hour limit stops it, run it again: finished runs are skipped."),
        md("## 1. Copy the code and the DrugBank file"), COPY, COPY_DB, CHECK_RISK,
        md("## 2. Install packages"), bash(PIP),
        md("## 3. Runs (live progress below)"),
        code("!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 "
             "--methods " + " ".join(f"harm_{m}_t{t.replace('0.', '')}_h{h.replace('0.', '')}" for m in ("loo", "solo")
                                     for t in ("0.005", "0.01", "0.02") for h in ("0.3", "0.5")) + " "
             "--attacks hideHigh hideHigh4 fabHigh moderate testB "
             "--device cuda --parallel 3 --out /kaggle/working/results/paperA_tune_harm"),
        md("## 4. Summary table"),
        code("!cd /kaggle/working/fedddi && python scripts/summarize_variants.py /kaggle/working/results/paperA_tune_harm"),
        md("## 5. Package results"), PACK],
}
# Step 7 (Paper A main results): fresh seeds 12-16, frozen settings (PUBLICATION_PLAN.md, "Frozen for step 7").
ALL9 = "fedavg normclip fltrust lfr zeno median krum caga harm_rep_d2"
SIX = "fedavg normclip fltrust lfr zeno caga"
MAIN = [  # (file, title, dataset, split, [(methods, attacks)], hours)
    ("01_db_core_a", "DrugBank random split: no attack, drift, two and four hiders", "drugbank", "random",
     [(ALL9, "clean drift hideHigh hideHigh4")], "6 to 7"),
    ("02_db_core_b", "DrugBank random split: fabrication, moderate poisoning, A Little Is Enough", "drugbank", "random",
     [(ALL9, "fabHigh moderate testB")], "5 to 6"),
    ("03_db_cold", "DrugBank cold-drug split: no attack, two hiders, A Little Is Enough", "drugbank", "cold",
     [(ALL9, "clean hideHigh testB")], "5 to 6"),
    ("04_db_model_a", "DrugBank random split: sign flip, Gaussian, IPM, min-max", "drugbank", "random",
     [(SIX, "signflip gauss ipm minmax")], "4 to 5"),
    ("05_db_model_b", "DrugBank random split: min-sum, strong poisoning, test A, hiding plus A Little Is Enough",
     "drugbank", "random", [(SIX, "minsum strong testA hideHighLIE")], "4 to 5"),
    ("06_ogbl_clean", "ogbl-ddi: no attack", "ogbl-ddi", "random", [(SIX, "clean")], "3 to 4"),
    ("07_ogbl_strong", "ogbl-ddi: strong poisoning", "ogbl-ddi", "random", [(SIX, "strong")], "3 to 4"),
    ("08_ogbl_testA", "ogbl-ddi: test A", "ogbl-ddi", "random", [(SIX, "testA")], "3 to 4"),
    ("09_ogbl_testB", "ogbl-ddi: A Little Is Enough", "ogbl-ddi", "random", [(SIX, "testB")], "3 to 4"),
    ("10_ogbl_minmax", "ogbl-ddi: min-max", "ogbl-ddi", "random", [(SIX, "minmax")], "3 to 4"),
]
# Step 8 (robustness and CAGA's candidate claims), same seeds and settings; criteria in PUBLICATION_PLAN.md.
DEF6 = "fedavg fltrust lfr zeno caga harm_rep_d2"
STEP8 = [
    ("01_adaptive_onoff", "adaptive hiders and the on-off hider", "drugbank", "random",
     [(DEF6 + " harm_rep_d2_open", "adaptive4"), ("harm_rep_d2_open", "adaptive4p clean hideHigh4"),
      (DEF6, "onoff4")], "4 to 5"),
    ("02_majority", "malicious majority (6 of 10) and under-estimated attacker counts", "drugbank", "random",
     [(DEF6 + " lfr_c2 zeno_c2 lfr_c4 zeno_c4", "hide6 lie6"), ("lfr_c2 zeno_c2", "hideHigh4 testB"),
      ("lfr_c4 zeno_c4", "hideHigh")], "5 to 6"),
    ("03_few", "one and two attackers", "drugbank", "random",
     [("fedavg fltrust caga lfr zeno lfr_c4 zeno_c4", "hide1 lie1 testB2"), ("lfr_c2 zeno_c2", "hide1 lie1")],
     "5 to 6"),
]
# Step 9 (ablations), tuning seeds 10-11; analysis fixed in PUBLICATION_PLAN.md before the runs.
ABL_ATK = "clean testB2 testB lie6 hideHigh4"
SRV_ATK = "clean hideHigh4 testB lie6"
STEP9 = [
    ("01_caga_parts", "CAGA's components and the probe size", "drugbank", "random",
     [("caga caga_w0 caga_tz1 caga_tz3 caga_block caga_cap", ABL_ATK),
      ("caga_p1k caga_p8k lfr_p1k lfr_p8k zeno_p1k zeno_p8k", SRV_ATK)], "4 to 5"),
    ("02_server_data", "how much clean data the server holds", "drugbank", "random",
     [("fedavg fltrust lfr zeno", SRV_ATK),
      ("fltrust_r05 fltrust_r20 lfr_r05 lfr_r20 zeno_r05 zeno_r20 caga_r05 caga_r20", SRV_ATK),
      ("anchor_h50_r05 anchor_h50_r20 server_h50_r05 server_h50_r20", SRV_ATK)], "5 to 6"),
]
# New method (after step 8): risk-anchored aggregation and FedVal, tuned on seeds 10-11; confirmation on fresh
# seeds 17-21 only after the pre-registered selection (PUBLICATION_PLAN.md, "New method").
NEW = [
    ("01_tune", "risk-anchored aggregation and FedVal, tuning", "drugbank", "random",
     [("fedavg fltrust fedavg_root fedval anchor_h25 anchor_h50 anchor_h100 anchor_h50_k05 anchor_avg_h50",
       "clean hideHigh hideHigh4 hide6 testB lie6 minmax")], "5 to 6"),
    ("02_controls", "server-only control and a smaller server share", "drugbank", "random",
     [("server_h25 server_h50 fltrust_r02 server_h25_r02 server_h50_r02 anchor_h25_r02 anchor_h50_r02",
       "clean hideHigh4 testB lie6")], "2.5 to 3"),
]
# Confirmation of risk-anchored aggregation on fresh seeds 17-21 (criteria in PUBLICATION_PLAN.md).
CONF9 = "fedavg fltrust lfr zeno fedval server_h25 server_h50 anchor_h25 anchor_h50"
CONFIRM = [
    ("01_core", "no attack and four hiders", "drugbank", "random", [(CONF9, "clean hideHigh4")], "4 to 5"),
    ("02_hiding", "two and six hiders, hiding plus A Little Is Enough", "drugbank", "random",
     [(CONF9, "hideHigh hide6 hideHighLIE")], "6 to 7"),
    ("03_general", "A Little Is Enough (4 and 6 attackers) and min-max", "drugbank", "random",
     [(CONF9 + " caga", "testB lie6 minmax")], "6 to 7"),
]
# Reviewer baselines for anchoring on tuning seeds 10-11 (FedAvg, FLTrust, anchor h50 and server-only h50 are
# reused from newmethod_01/02): client loss weighting, an enriched FLTrust root, a server fine-tune not fed back.
BASELINES = [
    ("01_baselines", "simpler ways to favour high-risk pairs", "drugbank", "random",
     [("fedavg_cw3 fltrust_cw3 fltrust_rh50 anchor_post_h50 finetune_h50 anchor_h50_cw3", "clean hideHigh4 lie6")],
     "2 to 2.5"),
]
# Anchoring plus client high-risk weighting, head-to-head on fresh seeds 22-26 (rule in PUBLICATION_PLAN.md).
CONFIRM_CW = [
    ("01_head_to_head", "anchoring with and without client high-risk weighting", "drugbank", "random",
     [("fltrust fltrust_cw3 anchor_h50 anchor_h50_cw3", "clean hideHigh4 lie6")], "2.5 to 3"),
]
# Newer defenses on the HARBOR confirmation seeds (17-21), joining that table; and an adaptive attacker built
# against HARBOR on fresh seeds 27-31 (criteria in PUBLICATION_PLAN.md).
NEWDEF = [
    ("01_defenses", "trimmed mean, centered clipping, DnC, FLAME, FreqFed", "drugbank", "random",
     [("trimmed cclip dnc flame freqfed", "clean hideHigh4 lie6 minmax")], "3 to 4"),
]
ADAPT = [
    ("01_trust_cone", "an adaptive attacker built against HARBOR", "drugbank", "random",
     [("fedavg fltrust fedval anchor_h50", "drift cone1 cone5"), ("server_h50", "drift")], "3 to 3.5"),
]
# Small server share confirmed on fresh seeds 32-36, and the second architecture (GCN) on the HARBOR
# confirmation seeds 17-21 (criteria in PUBLICATION_PLAN.md).
SHARE2 = [
    ("01_two_percent", "HARBOR with a 2% server share", "drugbank", "random",
     [("fedavg fltrust_r02 server_h50_r02 anchor_h50_r02", "clean hideHigh4 lie6")], "2 to 2.5"),
]
GCN = [
    ("01_gcn", "the same comparison with GCN layers", "drugbank", "random",
     [("fedavg_gcn fltrust_gcn server_h50_gcn anchor_h50_gcn", "clean hideHigh4 lie6")], "2 to 2.5"),
]
# Step 10, full scale (--scale full: 5x observations per client, 2x hidden size), fresh seeds 37-41; and GAT as the
# second architecture on the HARBOR confirmation seeds 17-21 (criteria in PUBLICATION_PLAN.md).
FULL = [
    ("01_core", "HARBOR, FLTrust, FedAvg and server-only, no attack and four hiders", "drugbank", "random",
     [("fedavg fltrust anchor_h50 server_h50", "clean hideHigh4")], "6 to 7"),
    ("02_lie6_central", "A Little Is Enough (6 attackers) and the centralized ceiling", "drugbank", "random",
     [("fedavg fltrust anchor_h50", "lie6"), ("central", "clean")], "3 to 4"),
    ("03_baselines_2pct", "baselines and a 2% server share", "drugbank", "random",
     [("fltrust_cw3 finetune_h50 fltrust_r02 anchor_h50_r02 server_h50_r02", "clean hideHigh4")], "7 to 8"),
]
GAT = [
    ("01_gat", "the same comparison with GAT layers", "drugbank", "random",
     [("fedavg_gat fltrust_gat server_h50_gat anchor_h50_gat", "clean hideHigh4 lie6")], "3 to 4"),
]
# GAT replication on fresh seeds 42-46 (2026-10-20): the first GAT set failed its informativeness check narrowly
# because GAT training varies across seeds; this independent set is judged on its own (PUBLICATION_PLAN.md).
GAT2 = [
    ("01_gat_replication", "GAT replication on fresh seeds", "drugbank", "random",
     [("fedavg_gat fltrust_gat server_h50_gat anchor_h50_gat", "clean hideHigh4 lie6")], "3"),
]
# F5 follow-up (2026-10-21): 2% share at full scale with the server sample matched to its data, fresh seeds 47-51.
FIX2 = [
    ("01_two_percent_matched", "2% server share at full scale, server sample matched to its data", "drugbank", "random",
     [("fltrust_r02_ro20 server_h50_r02_ro20 anchor_h50_r02_ro20 fltrust_r02", "clean hideHigh4")], "2 to 2.5"),
]
# Centralized ceiling redone (2026-10-21): the old "central" preset held one class-skewed sample, not all the data.
CEIL_DB = [("01_drugbank_full", "centralized ceiling on all training edges, DrugBank at full scale", "drugbank",
            "random", [("central_all", "clean")], "0.5")]
CEIL_OGB = [("01_ogbl_pilot", "centralized ceiling on all training edges, ogbl-ddi at pilot scale", "ogbl-ddi",
             "random", [("central_all", "clean")], "0.5 to 1")]
# Gap-closing checks before submission (2026-10-22; criteria in PUBLICATION_PLAN.md), pilot scale:
# HARBOR against the benchmark attacks it had not faced (seeds 52-56); local epochs, cap and enrichment
# (seeds 57-61); 80 rounds instead of 40 (seeds 57-61, so it pairs with the 40-round runs).
# All three in one notebook (about 9.5 GPU hours, under Kaggle's 12-hour limit), most important block first.
# A block may carry options: its own seeds, extra runner arguments and a subfolder of the notebook's results.
GAPS = [("01_all", "sensitivity, remaining attacks and 80 rounds", "drugbank", "random", [
    ("fedavg fltrust anchor_h50", "clean hideHigh4", {"seeds": "57 58 59 60 61", "sub": "sensitivity"}),
    ("fedavg_e1 fltrust_e1 anchor_h50_e1 fedavg_e5 fltrust_e5 anchor_h50_e5", "clean hideHigh4",
     {"seeds": "57 58 59 60 61", "sub": "sensitivity"}),
    ("anchor_h50_k05 anchor_h50_k2 anchor_h75", "clean hideHigh4", {"seeds": "57 58 59 60 61", "sub": "sensitivity"}),
    ("fedavg fltrust anchor_h50", "fabHigh strong signflip ipm minsum onoff4",
     {"seeds": "52 53 54 55 56", "sub": "more_attacks"}),
    ("fedavg fltrust anchor_h50", "clean hideHigh4",
     {"seeds": "57 58 59 60 61", "sub": "eighty_rounds", "extra": " --rounds 80"}),
], "9 to 10")]
# Second labelled dataset (2026-10-22): DDInter's expert grades (Major = high), DrugBank fingerprints; our DrugBank
# mapping failed validation against DDInter, so this becomes the primary safety analysis (PUBLICATION_PLAN.md).
DDI_CONF = "fedavg fltrust lfr zeno fedval server_h50 anchor_h25 anchor_h50"
DDINTER = [("01_confirm", "the confirmation on DDInter's expert severity grades", "drugbank", "random",
            [(DDI_CONF, "clean hideHigh hideHigh4 hide6 lie6 minmax")], "6 to 8")]
EXTRA = {"10": " --scale full", "F2": " --scale full", "CD": " --scale full", "DI": " --db-dir ddinter_graph"}
SEEDS = {"7": "12 13 14 15 16", "8": "12 13 14 15 16", "9": "10 11", "NM": "10 11", "NC": "17 18 19 20 21",
         "RB": "10 11", "CW": "22 23 24 25 26", "ND": "17 18 19 20 21",
         "AD": "27 28 29 30 31", "S2": "32 33 34 35 36", "GC": "17 18 19 20 21",
         "10": "37 38 39 40 41", "GA": "17 18 19 20 21",
         "GR": "42 43 44 45 46", "F2": "47 48 49 50 51",
         "CD": "37 38 39 40 41", "CO": "12 13 14 15 16",
         "GP": "52-61", "DI": "62 63 64 65 66"}
LABEL = {"7": "step 7", "8": "step 8", "9": "step 9", "NM": "new method", "NC": "new method confirmation",
         "RB": "anchoring baselines", "CW": "client-weighting confirmation",
         "ND": "newer defenses", "AD": "adaptive attack on HARBOR",
         "S2": "small server share confirmation", "GC": "second architecture",
         "10": "step 10 (full scale)", "GA": "second architecture (GAT)", "GR": "GAT replication", "F2": "2% share follow-up (full scale)",
         "CD": "centralized ceiling (DrugBank)", "CO": "centralized ceiling (ogbl-ddi)",
         "GP": "checks before submission", "DI": "DDInter"}
JOBS = ([("7", "main", len(MAIN), *m) for m in MAIN] + [("8", "s8", len(STEP8), *m) for m in STEP8]
        + [("9", "s9", len(STEP9), *m) for m in STEP9] + [("NM", "nm", len(NEW), *m) for m in NEW]
        + [("NC", "nc", len(CONFIRM), *m) for m in CONFIRM] + [("RB", "rb", len(BASELINES), *m) for m in BASELINES]
        + [("CW", "cw", len(CONFIRM_CW), *m) for m in CONFIRM_CW]
        + [("ND", "nd", len(NEWDEF), *m) for m in NEWDEF] + [("AD", "ad", len(ADAPT), *m) for m in ADAPT]
        + [("S2", "s2", len(SHARE2), *m) for m in SHARE2] + [("GC", "gc", len(GCN), *m) for m in GCN]
        + [("10", "s10", len(FULL), *m) for m in FULL] + [("GA", "ga", len(GAT), *m) for m in GAT]
        + [("GR", "gr", len(GAT2), *m) for m in GAT2] + [("F2", "f2", len(FIX2), *m) for m in FIX2]
        + [("CD", "cd", len(CEIL_DB), *m) for m in CEIL_DB] + [("CO", "co", len(CEIL_OGB), *m) for m in CEIL_OGB]
        + [("GP", "gp", 1, *m) for m in GAPS] + [("DI", "di", 1, *m) for m in DDINTER])
for step, prefix, total, key, title, dataset, split, blocks, hours in JOBS:
    out = f"/kaggle/working/results/{prefix}_{key}"
    opt = lambda b: b[2] if len(b) > 2 else {}  # noqa: E731
    n_runs = sum(len(b[0].split()) * len(b[1].split()) * len(opt(b).get("seeds", SEEDS[step]).split()) for b in blocks)
    db = dataset == "drugbank"
    cells = [md(f"# FedDDI-CAGA Paper A, {LABEL[step]} ({key[:2]} of {total}): {title}" + NL + NL + HEAD + NL + NL
                + ((PRIVATE + " **It must be the relabelled `drugbank_ddi.pt` with risk labels.**" + NL + NL) if db else "")
                + {"7": "Main results on fresh seeds 12 to 16 (never used before), frozen settings. ",
                   "8": "Robustness runs on seeds 12 to 16 with the step 7 settings. ",
                   "9": "Ablations on tuning seeds 10 and 11, step 7 settings otherwise. ",
                   "NM": "Tuning runs on seeds 10 and 11 (fresh seeds are kept for the confirmation), step 7 settings "
                         "otherwise. ",
                   "NC": "Confirmation on fresh seeds 17 to 21 (never used before), step 7 settings. ",
                   "RB": "Tuning seeds 10 and 11, step 7 settings; the comparison rows come from the new-method "
                         "notebooks. ",
                   "CW": "Confirmation on fresh seeds 22 to 26 (never used before), step 7 settings. ",
                   "ND": "Seeds 17 to 21 (the HARBOR confirmation seeds, so the rows join that table), step 7 "
                         "settings. ",
                   "AD": "Fresh seeds 27 to 31 (never used before), step 7 settings. ",
                   "S2": "Fresh seeds 32 to 36 (never used before), step 7 settings with a 2% server share. ",
                   "GC": "Seeds 17 to 21 (paired with the GraphSAGE confirmation), step 7 settings with GCN "
                         "layers. ",
                   "10": "Full scale on fresh seeds 37 to 41 (never used before): 100k observations per client and "
                         "hidden size 256; everything else as step 7. ",
                   "GA": "Seeds 17 to 21 (paired with the GraphSAGE confirmation), step 7 settings with GAT "
                         "layers. ",
                   "GR": "Fresh seeds 42 to 46 (never used before), step 7 settings with GAT layers. ",
                   "F2": "Fresh seeds 47 to 51 (never used before), full scale as step 10, 2% server share. ",
                   "CD": "Seeds 37 to 41 (paired with step 10), full scale; one client trains on every training edge. ",
                   "CO": "Seeds 12 to 16 (paired with step 7's ogbl-ddi runs), pilot scale; one client trains on every "
                         "training edge. ",
                   "GP": "Three checks in one notebook, step 7 settings except where a block changes them: "
                         "local epochs, cap and enrichment (fresh seeds 57 to 61); the remaining attacks against HARBOR "
                         "(fresh seeds 52 to 56); and 80 rounds (seeds 57 to 61, paired with the 40-round runs). Results "
                         "go to one subfolder per check. ",
                   "DI": "DDInter (expert severity grades; Major = high-risk) on fresh seeds 62 to 66, step 7 settings, "
                         "HARBOR not re-tuned. Also needs `ddinter_ddi.pt` (built with scripts/prepare_ddinter.py) in your "
                         "private dataset. "}[step] + f"{n_runs} runs, about "
                f"{hours} hours. If a session is cut off, see step 3 to resume without repeating finished runs."),
             md("## 1. Copy the code" + (" and the DrugBank file" if db else "")), COPY]
    version = ("grep -q model_attack scripts/cpu_pilot.py" if step == "7"
               else "grep -q harm_rep_d2_open scripts/gpu_variant_study.py" if step == "8"
               else "grep -q finetune_h50 scripts/gpu_variant_study.py" if step in ("9", "RB", "CW")
               else "grep -q trust-cone-cos scripts/cpu_pilot.py" if step in ("ND", "AD")
               else "grep -q build_dense_gcn_adj src/models/encoder.py" if step in ("S2", "GC")
               else "grep -q build_dense_edge_counts src/models/encoder.py" if step in ("10", "GA", "GR")
               else "grep -q _ro20 scripts/gpu_variant_study.py" if step == "F2"
               else "grep -q all-edges scripts/cpu_pilot.py" if step in ("CD", "CO")
               else "grep -q anchor_h50_k2 scripts/gpu_variant_study.py" if step == "GP"
               else "grep -q db-dir scripts/cpu_pilot.py" if step == "DI"
               else "grep -q '\"server_h50\"' scripts/gpu_variant_study.py")
    if db:
        cells += [COPY_DB, *([COPY_DD] if step == "DI" else []), CHECK_RISK, md("## 2. Install packages and check the code version"),
                  bash(PIP, "cd /kaggle/working/fedddi", f"{version} || "
                       "{ echo 'old fedddi_code.zip: upload the latest one'; exit 1; }")]
    else:
        cells += [md("## 2. Install and download ogbl-ddi (~5 min)"), stage('setup'),
                  bash("cd /kaggle/working/fedddi", f"{version} || "
                       "{ echo 'old fedddi_code.zip: upload the latest one'; exit 1; }")]
    cells.append(md("## 3. Resume: copy runs from an earlier session, if attached" + NL + NL +
                    "If a previous run of this notebook was cut off, open it, use **Add Input → Notebook Output** to attach "
                    "that version's output here, and run again: its finished runs are copied in and skipped. Without "
                    "such an input this cell does nothing."))
    cells.append(bash(f"mkdir -p {out}",
                      f"for d in $(find /kaggle/input -type d -name '{prefix}_{key}' 2>/dev/null); do "
                      f"echo \"resuming from $d\"; cp -rn \"$d\"/. {out}/; done",
                      f"echo \"$(ls {out} | grep -c seed) run folders present\""))
    cells.append(md("## 4. Runs (live progress below)"))
    for b in blocks:
        methods, attacks, o = b[0], b[1], opt(b)
        bout = f"{out}/{o['sub']}" if "sub" in o else out
        cells.append(code(f"!cd /kaggle/working/fedddi && python scripts/gpu_variant_study.py --dataset {dataset} "
                          f"--split {split} --seeds {o.get('seeds', SEEDS[step])} --methods {methods} --attacks {attacks} "
                          f"--device cuda --parallel 3 --out {bout}{EXTRA.get(step, '')}{o.get('extra', '')}"))
    subs = sorted({opt(b)["sub"] for b in blocks if "sub" in opt(b)})
    cells += [md("## 5. Summary table"),
              *[code(f"!cd /kaggle/working/fedddi && python scripts/summarize_variants.py {out}/{sub}") for sub in subs],
              *([] if subs else [code(f"!cd /kaggle/working/fedddi && python scripts/summarize_variants.py {out}")]),
              md("## 6. Package results"), PACK]
    name = {"7": "main", "NM": "newmethod", "NC": "confirm_anchor", "RB": "anchor_baselines", "CW": "confirm_cw", "ND": "newdefenses", "AD": "adaptive_harbor", "S2": "share2",
            "GC": "gcn", "10": "step10", "GA": "gat", "GR": "gat", "F2": "step10_fix", "CD": "ceiling", "CO": "ceiling", "GP": "gaps", "DI": "ddinter"}.get(step, "step" + step)
    nbs[f"fedddi_{name}_{key}.ipynb"] = cells

for name, cells in nbs.items():
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}, open('results/kaggle/' + name, 'w'), indent=1)
    print('wrote', name)
