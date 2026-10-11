"""Small-scale CPU pilot comparing FedAvg, norm clipping, FLTrust and CAGA without Flower or a GPU.

Client training, attack simulation and aggregation use the repository's own code (FedDDIClient,
PoisoningSimulator, DataDriftSimulator, caga_round, fltrust_aggregate, clip_and_average). Only the
Flower/Ray orchestration is replaced by a sequential loop, and the model and data are scaled down.
Results are an early signal for deciding whether to spend GPU time, not results for the paper.

Example:
    python scripts/cpu_pilot.py --method caga --cond joint --seed 0 --out results/cpu_pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import journal_protocol as jp  # noqa: E402
from src.data.data_drift import DataDriftSimulator, FeatureDriftEvent  # noqa: E402
from src.data.dataset import get_edge_split, load_dataset  # noqa: E402
from src.data.features import compute_fixed_global_features  # noqa: E402
from src.data.local_observations import generate_client_observations  # noqa: E402
from src.data.poisoning import PoisonEvent, PoisoningSimulator  # noqa: E402
from src.training.caga import CAGAHyperparams, caga_round, make_probe_loss_fn  # noqa: E402
from src.training.caga_probe import compute_block_index_map, sample_probe_edges  # noqa: E402
from src.training.evaluate import evaluate  # noqa: E402
from src.training.fl_client import FedDDIClient  # noqa: E402
from src.training.fltrust import fltrust_aggregate  # noqa: E402
from src.training.robust_aggregation import clip_and_average  # noqa: E402
from src.training.trainer import build_client_trainer  # noqa: E402
from src.utils.utils import seed_everything  # noqa: E402

METHODS = ("fedavg", "normclip", "fltrust", "caga")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--cond", choices=tuple(jp.CONDITIONS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/cpu_pilot")
    p.add_argument("--data", default=os.path.join(ROOT, "data"))
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--attack-start", type=int, default=15)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--obs-per-client", type=int, default=20_000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--poison-strength", choices=("moderate", "strong"), default="moderate")
    p.add_argument("--root-frac", type=float, default=0.12)
    p.add_argument("--probe-size", type=int, default=4000)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def make_cfg(a: argparse.Namespace, save_dir: str) -> Any:
    return OmegaConf.create({
        "dataset": {"num_anchors": 0},
        "model": {"hidden_dim": a.hidden, "num_layers": 2, "dropout": 0.3, "normalize": False},
        "predictor": {"hidden_dim": a.hidden, "num_layers": 3, "dropout": 0.3},
        "training": {"batch_size": a.batch_size, "num_neg_samples": 1, "lr": a.lr, "weight_decay": 0.0,
                     "epochs": 1, "save_dir": save_dir, "grad_clip": 1.0},
        "perf": {"dense_adj": True, "fast_negatives": True},
    })


def load_features(data_root: str, edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    path = os.path.join(data_root, f"fixed_features_cache_n{num_nodes}_a32_s16_seed{seed}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    feats = compute_fixed_global_features(edge_index.cpu(), num_nodes, num_anchors=32, num_spectral=16, seed=seed)
    torch.save(feats, path)
    return feats


def build_events(a: argparse.Namespace):
    P = dict(jp.PROTOCOL, attack_start=a.attack_start, poison_strength=a.poison_strength)
    poison_specs, drift_specs = jp.build_specs(P)
    pk, dk = jp.CONDITIONS[a.cond]
    poison = [PoisonEvent(client_id=s["client_id"], start_round=s["start_round"], suppress=s.get("suppress", {}),
                          emerge=s.get("emerge", {}), kind=s.get("kind", "sudden"),
                          transition_rounds=s.get("transition_rounds", 10), update_scale=s.get("update_scale", 1.0))
              for s in poison_specs[pk]]
    drift = [FeatureDriftEvent(client_id=s["client_id"], start_round=s["start_round"], feature_dims=s["feature_dims"],
                               shift_magnitude=s["shift_magnitude"], noise_std=s["noise_std"],
                               target_node_frac=s["target_node_frac"], node_selection=s["node_selection"],
                               kind=s["kind"], transition_rounds=s.get("transition_rounds", 10))
             for s in drift_specs[dk]]
    return poison, drift


def mean_over(reports: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in reports if key in r]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    a = parse_args()
    torch.set_num_threads(a.threads)
    seed_everything(a.seed)
    device = torch.device("cpu")
    run_name = f"{a.cond}_{a.method}_seed{a.seed}"
    out_dir = os.path.join(a.out, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    data, evaluator = load_dataset(root=a.data)
    split = get_edge_split(root=a.data)
    num_nodes = int(data.num_nodes)
    graph = data.edge_index
    node_features = load_features(a.data, graph, num_nodes, a.seed)
    atc_path = os.path.join(a.data, "ogbl_ddi", "atc_mapping.pt")
    atc_mapping = torch.load(atc_path) if os.path.exists(atc_path) else None
    if atc_mapping is None:
        print("WARNING: no ATC mapping; poisoning 'emerge' has no effect.", flush=True)
    cfg = make_cfg(a, os.path.join(out_dir, "ckpt"))

    # Mirror main_fl.py: only CAGA and FLTrust reserve root edges.
    uses_root = a.method in ("caga", "fltrust")
    train_full = split["train"]["edge"]
    root_edges = None
    client_pool = train_full
    if uses_root:
        perm = torch.randperm(train_full.size(0), generator=torch.Generator().manual_seed(a.seed + 999))
        n_root = max(50, int(a.root_frac * train_full.size(0)))
        root_edges, client_pool = train_full[perm[:n_root]].clone(), train_full[perm[n_root:]].clone()

    datasets = generate_client_observations(train_edges=client_pool, num_clients=a.num_clients,
                                            atc_mapping=atc_mapping, observations_per_client=a.obs_per_client,
                                            dirichlet_alpha=0.5, seed=a.seed)

    def trainer_for(cid: int, edges: torch.Tensor):
        return build_client_trainer(cid=cid, graph=graph, node_features=node_features, train_edges=edges,
                                    valid_split=split["valid"], test_split=split["test"], num_nodes=num_nodes,
                                    evaluator=evaluator, cfg=cfg, device=device)

    poison_events, drift_events = build_events(a)
    poison_sim = PoisoningSimulator(initial_datasets=datasets, train_edges=train_full, atc_mapping=atc_mapping,
                                    events=poison_events, observations_per_client=a.obs_per_client,
                                    seed=a.seed) if poison_events else None
    drift_sim = DataDriftSimulator(node_features=node_features, num_clients=a.num_clients, events=drift_events,
                                   seed=a.seed) if drift_events else None
    clients = [FedDDIClient(cid=str(i), trainer=trainer_for(i, datasets[i].edges), local_epochs=a.local_epochs,
                            device=device, num_rounds=a.rounds, c_id_int=i, poisoning_simulator=poison_sim,
                            data_drift_simulator=drift_sim) for i in range(a.num_clients)]
    special = sorted({e.client_id for e in poison_events} | {e.client_id for e in drift_events})

    eval_trainer = trainer_for(-2, client_pool[:1000])
    eval_client = FedDDIClient(cid="-2", trainer=eval_trainer, local_epochs=1, device=device)
    root_trainer = root_client = block_map = None
    hp = CAGAHyperparams(probe_size=a.probe_size)
    if uses_root:
        root_trainer = trainer_for(-1, root_edges)
        root_client = FedDDIClient(cid="-1", trainer=root_trainer, local_epochs=a.local_epochs, device=device,
                                   num_rounds=a.rounds)
        block_map = compute_block_index_map(root_trainer.encoder, root_trainer.predictor)

    params = eval_client.get_parameters({})
    rows: List[Dict[str, Any]] = []
    csv_path = os.path.join(out_dir, "rounds.csv")
    t_start = time.time()
    for rnd in range(1, a.rounds + 1):
        t0 = time.time()
        fits = [c.fit(params, {"server_round": rnd}) for c in clients]
        client_arrays = [f[0] for f in fits]
        weights = [float(f[1]) for f in fits]
        row: Dict[str, Any] = {"round": rnd, "train_loss": float(np.mean([f[2]["loss"] for f in fits]))}

        if a.method == "fedavg":
            tot = sum(weights)
            new = [sum(w * arrs[k] for w, arrs in zip(weights, client_arrays)) / tot for k in range(len(params))]
        elif a.method == "normclip":
            new, rep = clip_and_average(params, client_arrays, weights, 1.5, rnd)
            row["clipped"] = json.dumps(rep.clipped_mask)
        elif a.method == "fltrust":
            # FLTrust rescales client updates to the server update's norm, so the server takes as many
            # optimizer steps as a client: it trains on a fresh client-sized sample of root edges.
            g = torch.Generator().manual_seed(a.seed * 100_003 + rnd)
            pick = torch.randint(0, root_edges.size(0), (a.obs_per_client,), generator=g)
            root_trainer.split["train"]["edge"] = root_edges[pick]
            server_arrays = root_client.fit(params, {"server_round": rnd})[0]
            new, rep = fltrust_aggregate(params, client_arrays, server_arrays, [str(i) for i in range(a.num_clients)], rnd)
            row["fltrust_trust"] = json.dumps([round(t, 3) for t in rep.trust])
        else:
            probe = sample_probe_edges(root_pos_edges=root_trainer.split["train"]["edge"], num_nodes=num_nodes,
                                       m=hp.probe_size, seed=rnd, device=device,
                                       exclude_edge_index=root_trainer.edge_index)
            loss_fn = make_probe_loss_fn(root_trainer, block_map, probe)
            new, rep = caga_round(params, client_arrays, [str(i) for i in range(a.num_clients)], weights,
                                  block_map, loss_fn, hp, rnd)
            row.update(rep.as_row_fields())
        params = [n.astype(p.dtype, copy=False) for n, p in zip(new, params)]

        eval_client.set_parameters(params)
        test = evaluate(eval_trainer.encoder, eval_trainer.predictor, graph, split["test"]["edge"],
                        split["test"]["edge_neg"], evaluator, a.batch_size, device)
        val = evaluate(eval_trainer.encoder, eval_trainer.predictor, graph, split["valid"]["edge"],
                       split["valid"]["edge_neg"], evaluator, a.batch_size, device)
        row["test_hits20"], row["val_hits20"] = test["Hits@20"], val["Hits@20"]
        reports = [clients[i].evaluate(params, {"server_round": rnd})[2] for i in special]
        for key, col in (("drift_region_Hits@20", "drift_region_hits20"),
                         ("non_drift_region_Hits@20", "non_drift_region_hits20"),
                         ("poison_attack_mean_prob", "poison_attack_mean_prob")):
            v = mean_over(reports, key)
            if v is not None:
                row[col] = v
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{run_name}] round {rnd}/{a.rounds} test={row['test_hits20']:.4f} val={row['val_hits20']:.4f} "
              f"loss={row['train_loss']:.4f} {row['seconds']}s", flush=True)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "round", k))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"[{run_name}] done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
