"""Run the CAGA-variant study (pilot protocol) several runs at a time, on GPU or CPU.

Methods: FedAvg, norm clipping, FLTrust, CAGA with and without the class-aware probe and reputation,
and the robust baselines LFR, ERR, Zeno, Krum, coordinate median and trimmed mean. Attacks: moderate
poisoning, four unboosted poisoners ("test A"), A Little Is Enough ("test B"), and the safety-critical
attacks of Paper A (hiding or fabricating high-risk interactions, DrugBank only), each with severe drift. A run whose rounds.csv already has every round is skipped, so an
interrupted session can be restarted with the same command.

Example (Kaggle, DrugBank, tuning seeds):
    python scripts/gpu_variant_study.py --dataset drugbank --seeds 10 11 --device cuda --parallel 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = {
    "fedavg": ["--method", "fedavg"],
    "normclip": ["--method", "normclip", "--clip-multiplier", "1.0"],
    "fltrust": ["--method", "fltrust", "--root-obs", "20000"],
    "caga": ["--method", "caga_noblock", "--caga-warmup", "5"],
    "caga_class": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-class-aware"],
    "caga_rep": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-reputation"],
    "caga_rep3": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-reputation", "--caga-rep-threshold", "3"],
    "caga_class_rep": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-class-aware", "--caga-reputation"],
    # Calibrated eligibility (fixes false alarms on honest non-IID clients): outlier gate, plus warm-up baseline.
    "caga_out": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-outlier"],
    "caga_cal": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-outlier", "--caga-baseline"],
    # The same with the final check reduced to the divergence cap (caga_cal stalls once nobody is gated).
    "caga_cal2": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-outlier", "--caga-baseline",
                  "--caga-final-check", "cap"],
    # Harm-gated CAGA (Paper A, step 5): risk-stratified secret probe (half high-risk), CCH tolerance 0.01.
    # "loo" is the plan's leave-one-out CCH; "solo" judges each client's own model (not masked by other
    # attackers) and calibrates it like caga_cal, so honest specialists' steady offset does not count.
    "caga_harm": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-harm-gate", "--caga-harm-mode", "loo",
                  "--caga-secret-probe", "--caga-final-check", "cap"],
    "caga_harm_solo": ["--method", "caga_noblock", "--caga-warmup", "5", "--caga-harm-gate", "--caga-harm-mode", "solo",
                       "--caga-secret-probe", "--caga-outlier", "--caga-baseline", "--caga-final-check", "cap"],
    # Defenses that need the number of attackers, given a fixed guess instead of the true count (the
    # unnamed versions get the true count, an oracle the defender does not have).
    "lfr_c2": ["--method", "lfr", "--byz-c", "2"],
    "lfr_c4": ["--method", "lfr", "--byz-c", "4"],
    "zeno_c2": ["--method", "zeno", "--byz-c", "2"],
    "zeno_c4": ["--method", "zeno", "--byz-c", "4"],
    "krum_c2": ["--method", "krum", "--byz-c", "2"],
    # Paper A baselines (step 4); LFR, ERR and Zeno use CAGA's clean probe.
    "lfr": ["--method", "lfr"],
    "err": ["--method", "err"],
    "zeno": ["--method", "zeno"],
    "krum": ["--method", "krum"],
    "median": ["--method", "median"],
    "trimmed": ["--method", "trimmed"],
}
# Harm reputation (step 6d): solo harm, running mean per client, gated above (1 + delta) x the median
# client's; no cross-client calibration; cap-only final check.
for _d in ("0.1", "0.2", "0.3"):
    METHODS[f"harm_rep_d{_d.replace('0.', '')}"] = [
        "--method", "caga_noblock", "--caga-warmup", "5", "--caga-harm-gate", "--caga-harm-mode", "solo",
        "--caga-harm-tol", "0", "--caga-secret-probe", "--caga-final-check", "cap",
        "--caga-harm-rep", "--caga-harm-rep-delta", _d]

# Step 6 grid (Paper A): harm tolerance x high-risk share of the probe, for both harm modes. Names avoid dots
# so summarize_variants.py can parse them: t005 = 0.005, h3 = 0.3.
HARM_GRID = [(tol, hf) for tol in ("0.005", "0.01", "0.02") for hf in ("0.3", "0.5")]
for _mode in ("loo", "solo"):
    for _tol, _hf in HARM_GRID:
        METHODS[f"harm_{_mode}_t{_tol.replace('0.', '')}_h{_hf.replace('0.', '')}"] = (
            METHODS["caga_harm" if _mode == "loo" else "caga_harm_solo"]
            + ["--caga-harm-tol", _tol, "--caga-probe-high-frac", _hf])

ATTACKS = {
    "moderate": ["--cond", "joint"],
    "testA": ["--cond", "joint", "--poison-strength", "strong", "--extra-poisoned", "4", "5", "--no-boost"],
    "testB": ["--cond", "drift", "--lie-clients", "0", "1", "4", "5"],
    "strong": ["--cond", "joint", "--poison-strength", "strong"],
    "clean": ["--cond", "clean"],
    # Stress tests of A Little Is Enough: fewer attackers, and a larger shift than the paper's z.
    "testB2": ["--cond", "drift", "--lie-clients", "0", "1"],
    "testBz2": ["--cond", "drift", "--lie-clients", "0", "1", "4", "5", "--lie-z", "2.0"],
    # Drift only: the control that isolates the safety-critical attacks from the severe drift they run with.
    "drift": ["--cond", "drift"],
    # Paper A, step 3 (DrugBank): clients 0 and 1 hide 80% of their high-risk interactions, normal-size updates.
    "hideHigh": ["--cond", "joint", "--hide-high", "0.8", "--no-boost"],
    "hideHigh4": ["--cond", "joint", "--hide-high", "0.8", "--no-boost", "--extra-poisoned", "4", "5"],
    "fabHigh": ["--cond", "joint", "--fabricate-high", "0.3", "--no-boost"],
    "hideHighLIE": ["--cond", "joint", "--hide-high", "0.8", "--no-boost", "--lie-clients", "4", "5"],
    # Attacker-count study: the hiding attack with 1 or 6 attackers (hideHigh = 2, hideHigh4 = 4), and
    # A Little Is Enough with 1 attacker (testB2 = 2, testB = 4).
    "hide1": ["--cond", "joint", "--hide-high", "0.8", "--no-boost", "--poisoned-count", "1"],
    "hide6": ["--cond", "joint", "--hide-high", "0.8", "--no-boost", "--extra-poisoned", "4", "5", "6", "7"],
    "lie1": ["--cond", "drift", "--lie-clients", "0"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=("drugbank", "ogbl-ddi"), default="drugbank")
    p.add_argument("--split", default="random")
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--attacks", nargs="+", default=["moderate", "testA", "testB"], choices=list(ATTACKS))
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "gpu_variants"))
    p.add_argument("--log-cch", action="store_true", help="Pass --log-cch to every run.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def complete(run_dir: str, rounds: int) -> bool:
    f = os.path.join(run_dir, "rounds.csv")
    if not os.path.exists(f):
        return False
    with open(f) as fh:
        return sum(1 for _ in fh) - 1 >= rounds


def main() -> None:
    a = parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    jobs: List[Tuple[str, List[str]]] = []
    for seed in a.seeds:
        for atk in a.attacks:
            for m in a.methods:
                tag = f"{atk}_{m}"
                cmd = [sys.executable, os.path.join(ROOT, "scripts", "cpu_pilot.py"), "--dataset", a.dataset,
                       "--split", a.split, "--seed", str(seed), "--rounds", str(a.rounds), "--attack-start", "12",
                       "--lr", "0.003", "--local-epochs", "2", "--device", a.device, "--threads", str(a.threads),
                       "--tag", tag, "--out", a.out] + ATTACKS[atk] + METHODS[m] + (["--log-cch"] if a.log_cch else [])
                jobs.append((f"seed{seed}_{tag}", cmd))
    if a.dataset == "ogbl-ddi" and not a.dry_run:
        # Fixed features are cached per seed; build them once before parallel runs could race on the cache.
        for seed in a.seeds:
            subprocess.check_call([sys.executable, "-c", (
                "import sys; sys.path.insert(0, 'scripts'); sys.argv = ['x']; import cpu_pilot as c; "
                "from src.data.dataset import load_dataset; d, _ = load_dataset(root='data'); "
                f"c.load_features('data', d.edge_index, int(d.num_nodes), {seed})")], cwd=ROOT)
    todo = []
    for name, cmd in jobs:
        # cpu_pilot names its folder <cond>_<method>_seed<seed>_<tag>; the tag and seed identify it.
        seed, tag = name.split("_", 1)
        hits = [d for d in os.listdir(a.out) if d.endswith(f"_{seed}_{tag}")] if os.path.isdir(a.out) else []
        if hits and complete(os.path.join(a.out, hits[0]), a.rounds):
            continue
        todo.append((name, cmd))
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already complete, {len(todo)} to run", flush=True)
    if a.dry_run:
        for name, cmd in todo:
            print(name, " ".join(cmd[2:]))
        return

    def run(job: Tuple[str, List[str]]) -> Tuple[str, int, float]:
        name, cmd = job
        t0 = time.time()
        with open(os.path.join(a.out, "logs", f"{name}.log"), "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        return name, code, time.time() - t0

    done = 0
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for name, code, sec in pool.map(run, todo):
            done += 1
            status = "ok" if code == 0 else f"FAILED (exit {code}, see logs/{name}.log)"
            print(f"[{done}/{len(todo)}] {name}: {status} in {sec / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
