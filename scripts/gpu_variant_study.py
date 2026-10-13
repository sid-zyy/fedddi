"""Run the CAGA-variant study (pilot protocol) several runs at a time, on GPU or CPU.

Methods: FedAvg, norm clipping, FLTrust, and CAGA with and without the class-aware probe and
reputation. Attacks: moderate poisoning, four unboosted poisoners ("test A"), and A Little Is Enough
("test B"), each with severe drift. A run whose rounds.csv already has every round is skipped, so an
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
}
ATTACKS = {
    "moderate": ["--cond", "joint"],
    "testA": ["--cond", "joint", "--poison-strength", "strong", "--extra-poisoned", "4", "5", "--no-boost"],
    "testB": ["--cond", "drift", "--lie-clients", "0", "1", "4", "5"],
    "strong": ["--cond", "joint", "--poison-strength", "strong"],
    "clean": ["--cond", "clean"],
    # Stress tests of A Little Is Enough: fewer attackers, and a larger shift than the paper's z.
    "testB2": ["--cond", "drift", "--lie-clients", "0", "1"],
    "testBz2": ["--cond", "drift", "--lie-clients", "0", "1", "4", "5", "--lie-z", "2.0"],
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
                       "--tag", tag, "--out", a.out] + ATTACKS[atk] + METHODS[m]
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
