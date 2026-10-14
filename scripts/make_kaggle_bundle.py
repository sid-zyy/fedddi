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
            "nvidia-smi --query-gpu=name,memory.total --format=csv")
COPY_DB = bash("DB=$(find /kaggle/input -name drugbank_ddi.pt | head -1)",
               "[ -n \"$DB\" ] || { echo 'drugbank_ddi.pt not found: attach your private DrugBank dataset'; exit 1; }",
               "mkdir -p /kaggle/working/fedddi/data/drugbank && cp \"$DB\" /kaggle/working/fedddi/data/drugbank/",
               "echo \"copied $DB\"")
PIP = 'pip install -q torch-geometric ogb "flwr[simulation]>=1.10" hydra-core omegaconf'
PACK = code(NL.join(["import shutil", "shutil.make_archive('/kaggle/working/results', 'zip', '/kaggle/working/results')",
                     "print('Download results.zip from the Output tab')"]))
def stage(name): return bash(f"cd /kaggle/working/fedddi && bash scripts/kaggle_gpu_checks.sh {name}")
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
        md("## 3. Centralized, fixed structural features, 3 runs (the ceiling for federated scores)"), stage('centralized'),
        md("## 4. Centralized, learned embeddings, 1 run (compare with the OGB leaderboard)"), stage('centralized_emb'),
        md("## 5. Package results"), PACK],
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
}
for name, cells in nbs.items():
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}, open('results/kaggle/' + name, 'w'), indent=1)
    print('wrote', name)
