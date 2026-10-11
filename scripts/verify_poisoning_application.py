#!/usr/bin/env python3
"""Sanity check for data poisoning, run BEFORE trusting (or distrusting) any downstream performance
numbers
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.dataset import get_edge_split
from src.data.local_observations import generate_client_observations
from src.data.poisoning import PoisonEvent, PoisoningSimulator

POISON_SPECS = {
    "mild": [
        PoisonEvent(client_id=0, start_round=25, kind="sudden",
                    suppress={0: 0.30}, emerge={1: 0.20}),
    ],
    "light": [





        PoisonEvent(client_id=0, start_round=10, kind="sudden",
                    suppress={0: 0.10}, emerge={1: 0.10}, update_scale=2.5),
    ],
    "moderate": [
        PoisonEvent(client_id=0, start_round=16, kind="sudden",
                    suppress={0: 0.50}, emerge={1: 0.30}),
        PoisonEvent(client_id=1, start_round=20, kind="gradual", transition_rounds=10,
                    suppress={2: 0.40}, emerge={0: 0.40}, update_scale=2.0),
    ],
    "severe": [
        PoisonEvent(client_id=0, start_round=16, kind="sudden",
                    suppress={0: 0.70}, emerge={1: 0.60}, update_scale=3.0),
        PoisonEvent(client_id=1, start_round=16, kind="sudden",
                    suppress={2: 0.70}, emerge={0: 0.60}, update_scale=3.0),
        PoisonEvent(client_id=2, start_round=16, kind="sudden",
                    suppress={1: 0.70}, emerge={0: 0.60}, update_scale=3.0),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", choices=list(POISON_SPECS), default="moderate")
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", default="1,15,16,17,20,22,25,30,40,50")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--atc-mapping", default="./data/ogbl_ddi/atc_mapping.pt")
    args = ap.parse_args()

    checkpoint_rounds = [int(r) for r in args.rounds.split(",")]
    events = POISON_SPECS[args.severity]
    malicious_clients = sorted({ev.client_id for ev in events})
    honest_clients = [c for c in range(args.num_clients) if c not in malicious_clients]

    print(f"Severity spec: {args.severity}")
    for ev in events:
        print(f"  client {ev.client_id}: {ev.kind} attack @ round {ev.start_round} -> "
              f"suppress={ev.suppress} emerge={ev.emerge} update_scale={ev.update_scale}")
    print(f"Malicious clients: {malicious_clients}   Honest (control) clients: {honest_clients}\n")

    print("Loading edge split and ATC mapping ...")
    split = get_edge_split(root=args.data_dir)
    train_edges = split["train"]["edge"]
    try:
        atc_mapping = torch.load(args.atc_mapping, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: {args.atc_mapping} not found. Run scripts/prepare_atc_mapping.py first.")
        sys.exit(1)

    initial_datasets = generate_client_observations(
        train_edges, args.num_clients, atc_mapping=atc_mapping,
        dirichlet_alpha=args.dirichlet_alpha, seed=args.seed,
    )
    n_obs = len(initial_datasets[0].edges)
    print(f"n_obs per client per round: {n_obs}\n")

    sim = PoisoningSimulator(
        initial_datasets, train_edges, atc_mapping,
        events=events, seed=args.seed,
    )

    rows = []
    for r in checkpoint_rounds:
        datasets = sim.step(r)
        for cid in range(args.num_clients):
            ds = datasets[cid]
            rows.append({
                "round": r, "client": cid,
                "role": "MALICIOUS" if cid in malicious_clients else "honest",
                "false_negative": int(ds.false_negative_edges.size(0)),
                "emerged": int(ds.emerged_edges.size(0)),
                "update_scale": sim.update_scale(cid, r),
            })

    print(f"{'Round':>5} {'Client':>6} {'Role':>10} {'false_neg':>10} {'emerged':>8} {'update_scale':>13}")
    print("-" * 62)
    for row in rows:
        print(f"{row['round']:>5} {row['client']:>6} {row['role']:>10} "
              f"{row['false_negative']:>10} {row['emerged']:>8} {row['update_scale']:>13.2f}")


    print("\n" + "=" * 70)
    last_round = checkpoint_rounds[-1]
    ok = True
    for cid in malicious_clients:
        row = next(r for r in rows if r["round"] == last_round and r["client"] == cid)
        corrupted = row["false_negative"] + row["emerged"]
        status = "PASS" if corrupted > 0 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"Client {cid} (should be corrupted): "
              f"false_neg={row['false_negative']} emerged={row['emerged']} "
              f"[{status}]")
    for cid in honest_clients:
        row = next(r for r in rows if r["round"] == last_round and r["client"] == cid)
        corrupted = row["false_negative"] + row["emerged"]
        status = "PASS" if corrupted == 0 else "WARN"
        print(f"Client {cid} (control, honest): "
              f"false_neg={row['false_negative']} emerged={row['emerged']} "
              f"[{status}, should be exactly 0]")

    print("=" * 70)
    if ok:
        print("VERDICT: poisoning IS measurably injecting corrupted edges into every")
        print("configured malicious client's training data. If downstream model/attack-")
        print("success metrics still don't move, the corruption itself is not the")
        print("missing piece -- check whether the attack survives FedAvg dilution")
        print("(increase fraction_malicious and/or update_scale), and check")
        print("poison_attack_mean_prob in the per-round metrics (src/training/")
        print("fl_client.py:evaluate) rather than only the global clean Hits@20.")
    else:
        print("VERDICT: at least one malicious client's returned dataset had zero")
        print("corrupted edges at the final checkpoint. That means the attack isn't")
        print("reaching the training data as configured -- fix this before drawing any")
        print("conclusion about poisoning's effect on model performance. Check: is")
        print("start_round/transition_rounds within your num_rounds budget? Does the")
        print("target ATC class have matching edges (suppress) / node population")
        print("(emerge) in this dataset?")
        sys.exit(1)


if __name__ == "__main__":
    main()
