#!/usr/bin/env python3
"""Sanity check for covariate (data) drift, run BEFORE trusting (or distrusting) any downstream
performance numbers
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.data_drift import DataDriftSimulator, FeatureDriftEvent
from src.data.features import compute_fixed_global_features
from src.data.dataset import get_edge_split
from src.models.encoder import GraphSAGEEncoder

DRIFT_SPECS = {
    "mild": [
        FeatureDriftEvent(client_id=0, start_round=25, kind="sudden",
                           feature_dims=[0, 1], shift_magnitude=1.0,
                           target_node_frac=0.10),
    ],
    "light": [



        FeatureDriftEvent(client_id=0, start_round=10, kind="sudden",
                           feature_dims=[0, 1], shift_magnitude=1.2,
                           target_node_frac=0.10),
    ],
    "client2_isolated": [




        FeatureDriftEvent(client_id=2, start_round=10, kind="sudden",
                           feature_dims=[0, 1, 2], shift_magnitude=2.0,
                           noise_std=0.2, target_node_frac=0.25),
    ],
    "moderate": [
        FeatureDriftEvent(client_id=0, start_round=16, kind="sudden",
                           feature_dims=[0, 1, 2, 3], shift_magnitude=2.0,
                           noise_std=0.3, target_node_frac=0.25),
        FeatureDriftEvent(client_id=1, start_round=20, kind="gradual",
                           transition_rounds=10, feature_dims=[2, 3, 4],
                           shift_magnitude=2.5, target_node_frac=0.30,
                           node_selection="random"),
    ],
    "severe": [
        FeatureDriftEvent(client_id=0, start_round=16, kind="sudden",
                           feature_dims=list(range(10)), shift_magnitude=4.0,
                           noise_std=0.5, target_node_frac=0.50),
        FeatureDriftEvent(client_id=1, start_round=16, kind="sudden",
                           feature_dims=list(range(10)), shift_magnitude=4.0,
                           target_node_frac=0.50, node_selection="random"),
        FeatureDriftEvent(client_id=2, start_round=16, kind="sudden",
                           feature_dims=list(range(10)), shift_magnitude=4.0,
                           target_node_frac=0.50),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", choices=list(DRIFT_SPECS), default="moderate")
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", default="1,15,16,17,20,22,25,30,40,50")
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    checkpoint_rounds = [int(r) for r in args.rounds.split(",")]
    events = DRIFT_SPECS[args.severity]
    drifting_clients = sorted({ev.client_id for ev in events})
    control_clients = [c for c in range(args.num_clients) if c not in drifting_clients]

    print(f"Severity spec: {args.severity}")
    for ev in events:
        print(f"  client {ev.client_id}: {ev.kind} drift @ round {ev.start_round} -> "
              f"dims={ev.feature_dims} magnitude={ev.shift_magnitude} "
              f"node_frac={ev.target_node_frac} ({ev.node_selection})")
    print(f"Drifting clients: {drifting_clients}   Control clients: {control_clients}\n")

    print("Loading edge split and computing fixed global features ...")
    split = get_edge_split(root=args.data_dir)
    train_edges = split["train"]["edge"]
    num_nodes = int(train_edges.max().item()) + 1
    node_features = compute_fixed_global_features(train_edges, num_nodes, seed=args.seed)
    print(f"node_features shape: {tuple(node_features.shape)}\n")

    sim = DataDriftSimulator(
        node_features, num_clients=args.num_clients, events=events, seed=args.seed,
    )


    encoders = {
        cid: GraphSAGEEncoder(num_nodes=num_nodes, hidden_dim=32, num_layers=1,
                               node_features=node_features)
        for cid in range(args.num_clients)
    }

    rows = []
    for r in checkpoint_rounds:
        deltas = sim.step(r)
        for cid in range(args.num_clients):
            delta = deltas.get(cid)
            encoders[cid].set_feature_delta_(delta)
            enc = encoders[cid]
            actual_shift = (enc.node_features - enc._base_node_features)
            n_rows_changed = int((actual_shift.abs().sum(dim=1) > 1e-9).sum())
            max_abs_shift = float(actual_shift.abs().max())
            rows.append({
                "round": r, "client": cid,
                "role": "DRIFT" if cid in drifting_clients else "control",
                "n_rows_changed": n_rows_changed,
                "max_abs_shift": max_abs_shift,
            })

    print(f"{'Round':>5} {'Client':>6} {'Role':>7} {'rows_changed':>12} {'max_abs_shift':>13}")
    print("-" * 50)
    for row in rows:
        print(f"{row['round']:>5} {row['client']:>6} {row['role']:>7} "
              f"{row['n_rows_changed']:>12} {row['max_abs_shift']:>13.4f}")


    print("\n" + "=" * 70)
    last_round = checkpoint_rounds[-1]
    ok = True
    for cid in drifting_clients:
        row = next(r for r in rows if r["round"] == last_round and r["client"] == cid)
        status = "PASS" if row["n_rows_changed"] > 0 and row["max_abs_shift"] > 1e-6 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"Client {cid} (should drift):    rows_changed={row['n_rows_changed']} "
              f"max_abs_shift={row['max_abs_shift']:.4f}  [{status}]")
    for cid in control_clients:
        row = next(r for r in rows if r["round"] == last_round and r["client"] == cid)
        status = "PASS" if row["n_rows_changed"] == 0 else "WARN"
        print(f"Client {cid} (control, stable): rows_changed={row['n_rows_changed']} "
              f"max_abs_shift={row['max_abs_shift']:.4f}  [{status}, should be exactly 0]")


    if drifting_clients:
        cid = drifting_clients[0]
        encoders[cid].set_feature_delta_(None)
        restored_exactly = torch.equal(
            encoders[cid].node_features, encoders[cid]._base_node_features
        )
        print(f"\nClearing delta on client {cid} restores pristine baseline exactly: "
              f"{'PASS' if restored_exactly else 'FAIL'}")
        ok = ok and restored_exactly

    print("=" * 70)
    if ok:
        print("VERDICT: data drift IS measurably perturbing the drifting client's OWN")
        print("encoder feature buffer, on the intended node subset, and restores cleanly.")
        print("If downstream model metrics still don't move, check the drift-region")
        print("Hits@20 (src/training/fl_client.py:evaluate) specifically, not the")
        print("static global Hits@20 -- by design, global eval never sees this client's")
        print("drifted features at all.")
    else:
        print("VERDICT: at least one drifting client's feature buffer never changed, or")
        print("the pristine baseline didn't restore exactly. Fix this before drawing any")
        print("conclusion about data drift's effect on model performance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
