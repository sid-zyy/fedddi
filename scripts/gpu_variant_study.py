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
    # Newer baselines (2026-10-19): centered clipping, Divide-and-Conquer (true attacker count), FLAME, FreqFed.
    "cclip": ["--method", "cclip"],
    "dnc": ["--method", "dnc"],
    "flame": ["--method", "flame"],
    "freqfed": ["--method", "freqfed"],
}
# Harm reputation (step 6d): solo harm, running mean per client, gated above (1 + delta) x the median
# client's; no cross-client calibration; cap-only final check.
for _d in ("0.1", "0.2", "0.3"):
    METHODS[f"harm_rep_d{_d.replace('0.', '')}"] = [
        "--method", "caga_noblock", "--caga-warmup", "5", "--caga-harm-gate", "--caga-harm-mode", "solo",
        "--caga-harm-tol", "0", "--caga-secret-probe", "--caga-final-check", "cap",
        "--caga-harm-rep", "--caga-harm-rep-delta", _d]
# Step 8: the same with a probe the clients can know (no secret re-draw), the target of `adaptive4p`.
METHODS["harm_rep_d2_open"] = [x for x in METHODS["harm_rep_d2"] if x != "--caga-secret-probe"]
# Step 9 ablations. CAGA's components: no warm-up, paired-test threshold 1 or 3 (default 2), per-block gating
# (convolutions / MLP / input map) instead of whole clients, final step-size check reduced to the cap.
METHODS.update({
    "caga_w0": ["--method", "caga_noblock", "--caga-warmup", "0"],
    "caga_tz1": METHODS["caga"] + ["--caga-tol-z", "1"],
    "caga_tz3": METHODS["caga"] + ["--caga-tol-z", "3"],
    "caga_block": ["--method", "caga", "--caga-warmup", "5"],
    "caga_cap": METHODS["caga"] + ["--caga-final-check", "cap"],
})
# Server data: probe size (default 4,000) for the probe defenses, and the share of training pairs the server
# holds (default 0.12) for every defense that uses server data (r05 = 0.05, r20 = 0.2).
for _m in ("caga", "lfr", "zeno"):
    METHODS[f"{_m}_p1k"] = METHODS[_m] + ["--probe-size", "1000"]
    METHODS[f"{_m}_p8k"] = METHODS[_m] + ["--probe-size", "8000"]
for _m in ("caga", "lfr", "zeno", "fltrust"):
    METHODS[f"{_m}_r05"] = METHODS[_m] + ["--root-frac", "0.05"]
    METHODS[f"{_m}_r20"] = METHODS[_m] + ["--root-frac", "0.2"]
# New-method study (after step 8): FedVal (per-risk-class validation scores; a baseline reviewers expect), the
# server as one more client (fedavg_root), and risk-anchored aggregation: FLTrust (or FedAvg) aggregate, then a
# server step from it on clean pairs with h = share of high-risk pairs (h25 ~ the root's own share, i.e. no
# enrichment: the "just more server training" control), capped at kappa x the aggregate's update norm.
METHODS["fedval"] = ["--method", "fedval"]
METHODS["fedavg_root"] = ["--method", "fedavg_root", "--root-obs", "20000"]
for _name, (_h, _k, _base) in {"anchor_h25": ("0.25", "1.0", "fltrust"), "anchor_h50": ("0.5", "1.0", "fltrust"),
                               "anchor_h100": ("1.0", "1.0", "fltrust"), "anchor_h50_k05": ("0.5", "0.5", "fltrust"),
                               "anchor_avg_h50": ("0.5", "1.0", "fedavg")}.items():
    METHODS[_name] = ["--method", "anchor", "--root-obs", "20000", "--anchor-base", _base,
                      "--anchor-high-frac", _h, "--anchor-kappa", _k]
# Controls after the tuning run: the server alone (same sampling, ignores the clients), and every server-data
# method with a smaller server share (2% of the training pairs instead of 12%).
for _name, _h in (("server_h25", "0.25"), ("server_h50", "0.5")):
    METHODS[_name] = ["--method", "anchor", "--root-obs", "20000", "--anchor-base", "none",
                                    "--anchor-high-frac", _h]
for _m in ("fltrust", "server_h25", "server_h50", "anchor_h25", "anchor_h50"):
    METHODS[f"{_m}_r02"] = METHODS[_m] + ["--root-frac", "0.02"]
for _m in ("server_h50", "anchor_h50"):
    METHODS[f"{_m}_r05"] = METHODS[_m] + ["--root-frac", "0.05"]
    METHODS[f"{_m}_r20"] = METHODS[_m] + ["--root-frac", "0.2"]
# Second architecture (2026-10-19): the same methods with GCN layers instead of GraphSAGE, settings unchanged.
for _m in ("fedavg", "fltrust", "server_h50", "anchor_h50"):
    METHODS[f"{_m}_gcn"] = METHODS[_m] + ["--conv", "gcn"]
# Reviewer baselines for anchoring (simpler ways to favour high-risk pairs). Weight 3 matches the h50 enrichment:
# high-risk pairs are ~25% of positives, so 50% enrichment raises their relative weight by (0.75/0.25) = 3.
# _cw3: honest clients up-weight high-risk positives in their loss; fltrust_rh50: FLTrust's server sample enriched;
# anchor_post_h50: the anchored model is deployed but not fed back; finetune_h50: the same, uncapped (a plain
# server fine-tune of each round's FLTrust model).
METHODS["fedavg_cw3"] = METHODS["fedavg"] + ["--client-high-weight", "3"]
METHODS["fltrust_cw3"] = METHODS["fltrust"] + ["--client-high-weight", "3"]
METHODS["fltrust_rh50"] = METHODS["fltrust"] + ["--fltrust-high-frac", "0.5"]
METHODS["anchor_post_h50"] = METHODS["anchor_h50"] + ["--anchor-feedback", "0"]
METHODS["finetune_h50"] = METHODS["anchor_h50"][:-2] + ["--anchor-kappa", "1000000", "--anchor-feedback", "0"]
METHODS["anchor_h50_cw3"] = METHODS["anchor_h50"] + ["--client-high-weight", "3"]

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
    # Step 7 model-poisoning attacks: clients 0, 1, 4 and 5 (as in test B), severe drift on 2 and 3.
    **{k: ["--cond", "drift", "--model-attack", k, "--model-attack-clients", "0", "1", "4", "5"]
       for k in ("signflip", "gauss", "ipm", "minmax", "minsum")},
    # Step 8: malicious majority with A Little Is Enough (fixed z = 0.84, test B's value for 4 attackers; the
    # attack's own formula is undefined for a majority), the on-off hider, and adaptive hiders that keep their
    # estimated harm under the harm-reputation limit, estimated on their own data or on the server's probe
    # (the latter only against a defense without a secret probe).
    "lie6": ["--cond", "drift", "--lie-clients", "0", "1", "4", "5", "6", "7", "--lie-z", "0.84"],
    # Adaptive attack against HARBOR (2026-10-19): 4 attackers (as in test B) send high-risk-suppressing updates
    # inside FLTrust's trust cone, at cosine 0.1 (maximal suppression) or 0.5 (more trust weight).
    **{f"cone{c}": ["--cond", "drift", "--model-attack-clients", "0", "1", "4", "5", "--trust-cone-cos", f"0.{c}"]
       for c in ("1", "5")},
    "onoff4": ["--cond", "joint", "--hide-high", "0.8", "--no-boost", "--extra-poisoned", "4", "5",
               "--on-off", "12-20,30-40"],
    "adaptive4": ["--cond", "joint", "--hide-high", "0.8", "--no-boost", "--extra-poisoned", "4", "5", "--adaptive-hide"],
    "adaptive4p": ["--cond", "joint", "--hide-high", "0.8", "--no-boost", "--extra-poisoned", "4", "5", "--adaptive-hide",
                   "--adaptive-knows-probe"],
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
