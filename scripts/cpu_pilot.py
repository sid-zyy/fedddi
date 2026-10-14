"""Small-scale CPU pilot comparing FedAvg, norm clipping, FLTrust, CAGA and other robust baselines
(LFR, ERR, Zeno, Krum, coordinate median, trimmed mean) without Flower or a GPU.

Client training, attack simulation and aggregation use the repository's own code (FedDDIClient,
PoisoningSimulator, DataDriftSimulator, caga_round, fltrust_aggregate, clip_and_average). Only the
Flower/Ray orchestration is replaced by a sequential loop, and the model and data are scaled down.
Results are an early signal for deciding whether to spend GPU time, not results for the paper.

Example:
    python scripts/cpu_pilot.py --method caga --cond joint --seed 0 --out results/cpu_pilot
"""
from __future__ import annotations

import argparse
import copy
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
from src.training.caga import pairwise_rank_loss  # noqa: E402
from src.training.caga_probe import score_arrays_on_probe  # noqa: E402
from src.training.evaluate import score_edges  # noqa: E402
from src.training.fl_client import FedDDIClient  # noqa: E402
from src.training.fltrust import fltrust_aggregate  # noqa: E402
from src.training.robust_aggregation import (  # noqa: E402
    clip_and_average, err_aggregate, krum_aggregate, lfr_aggregate, median_aggregate, probe_auroc_error,
    trimmed_mean_aggregate, zeno_aggregate)
from src.training import safety_metrics as sm  # noqa: E402
from src.training.trainer import build_client_trainer  # noqa: E402
from src.utils.utils import seed_everything  # noqa: E402

METHODS = ("fedavg", "fedavg_root", "normclip", "fltrust", "caga", "caga_noblock",
           "lfr", "err", "zeno", "krum", "median", "trimmed")
# Methods that reserve clean server ("root") edges: CAGA and FLTrust train or probe on them; LFR, ERR
# and Zeno score candidates on the same probe CAGA uses.
ROOT_METHODS = ("caga", "caga_noblock", "fltrust", "fedavg_root", "lfr", "err", "zeno")
PROBE_METHODS = ("caga", "caga_noblock", "lfr", "err", "zeno")
DRIFT_LEVELS = jp.DRIFT_LEVELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--cond", choices=tuple(jp.CONDITIONS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drift-level", choices=tuple(DRIFT_LEVELS), default=jp.PROTOCOL["drift_strength"])
    p.add_argument("--out", default="results/cpu_pilot")
    p.add_argument("--data", default=os.path.join(ROOT, "data"))
    p.add_argument("--dataset", choices=("ogbl-ddi", "drugbank"), default="ogbl-ddi")
    p.add_argument("--split", choices=("random", "cold"), default="random", help="DrugBank split.")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--attack-start", type=int, default=15)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--obs-per-client", type=int, default=20_000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--poison-strength", choices=("moderate", "strong"), default="moderate")
    p.add_argument("--extra-poisoned", type=int, nargs="*", default=[],
                   help="More poisoned clients; each copies the protocol's poison events in turn.")
    p.add_argument("--no-boost", action="store_true", help="Poisoned clients send normal-size updates.")
    p.add_argument("--round-lr-decay", action="store_true", help="Cosine learning-rate decay across rounds.")
    p.add_argument("--lie-clients", type=int, nargs="*", default=[],
                   help="Clients running the 'A Little Is Enough' attack from --attack-start onwards.")
    p.add_argument("--lie-z", type=float, default=None,
                   help="Override the attack's z (default: the paper's value for the number of attackers).")
    p.add_argument("--hide-high", type=float, default=0.0,
                   help="Poisoned clients drop this share of their own high-risk positives (DrugBank); "
                        "replaces the class-based suppress/emerge attack.")
    p.add_argument("--fabricate-high", type=float, default=0.0,
                   help="Poisoned clients add this share of fake interactions between drugs from high-risk "
                        "ATC class pairs (DrugBank); replaces the class-based suppress/emerge attack.")
    p.add_argument("--byz-c", type=int, default=None,
                   help="Attackers assumed by LFR/ERR (c), Zeno (b) and Krum (f). Default: the true number of "
                        "attackers, or 2 when there are none.")
    p.add_argument("--zeno-rho", type=float, default=0.0005)
    p.add_argument("--trim", type=float, default=0.2, help="Trimmed mean: share dropped from each end.")
    p.add_argument("--log-cch", action="store_true",
                   help="Log every client's Counterfactual Clinical Harm each round (leave-one-out).")
    p.add_argument("--cch-anchor", type=int, default=2000,
                   help="High-risk validation positives (and as many validation negatives) in the CCH anchor.")
    p.add_argument("--root-frac", type=float, default=0.12)
    p.add_argument("--probe-size", type=int, default=4000)
    p.add_argument("--caga-tol-z", type=float, default=2.0)
    p.add_argument("--caga-tol-rel", type=float, default=0.0)
    p.add_argument("--caga-warmup", type=int, default=0)
    p.add_argument("--caga-class-aware", action="store_true", help="Judge each client on its own drug classes.")
    p.add_argument("--caga-reputation", action="store_true", help="Gate only clients with a bad running record.")
    p.add_argument("--caga-rep-beta", type=float, default=0.7)
    p.add_argument("--caga-rep-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cpu", help="cpu or cuda; GPU runs use the same protocol.")
    p.add_argument("--clip-multiplier", type=float, default=1.5)
    p.add_argument("--root-obs", type=int, default=None,
                   help="Root edges the server trains on per round (FLTrust, fedavg_root); default: --obs-per-client.")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tag", default="", help="Suffix for the run name, e.g. a hyperparameter setting.")
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


def build_events(a: argparse.Namespace, num_features: int = 50):
    P = dict(jp.PROTOCOL, attack_start=a.attack_start, poison_strength=a.poison_strength,
             drift_strength=a.drift_level)
    poison_specs, drift_specs = jp.build_specs(P)
    pk, dk = jp.CONDITIONS[a.cond]
    poison = [PoisonEvent(client_id=s["client_id"], start_round=s["start_round"], suppress=s.get("suppress", {}),
                          emerge=s.get("emerge", {}), kind=s.get("kind", "sudden"),
                          transition_rounds=s.get("transition_rounds", 10), update_scale=s.get("update_scale", 1.0))
              for s in poison_specs[pk]]
    if a.hide_high or a.fabricate_high:
        # Safety-critical poisoning: the same clients and timing, but they target high-risk interactions.
        for e in poison:
            e.suppress, e.emerge = {}, {}
            e.hide_risk, e.fabricate_high = a.hide_high, a.fabricate_high
    if poison:
        base = list(poison)
        taken = {e.client_id for e in base} | {s["client_id"] for s in drift_specs[dk]}
        for i, cid in enumerate(a.extra_poisoned):
            if cid in taken:
                raise ValueError(f"client {cid} is already poisoned or drifted")
            src = base[i % len(base)]
            poison.append(PoisonEvent(client_id=cid, start_round=src.start_round, suppress=dict(src.suppress),
                                      emerge=dict(src.emerge), kind=src.kind, transition_rounds=src.transition_rounds,
                                      update_scale=src.update_scale, hide_risk=src.hide_risk,
                                      fabricate_high=src.fabricate_high))
            taken.add(cid)
        if a.no_boost:
            for e in poison:
                e.update_scale = 1.0
    drift = [FeatureDriftEvent(client_id=s["client_id"], start_round=s["start_round"], feature_dims=s["feature_dims"],
                               shift_magnitude=s["shift_magnitude"], noise_std=s["noise_std"],
                               target_node_frac=s["target_node_frac"], node_selection=s["node_selection"],
                               kind=s["kind"], transition_rounds=s.get("transition_rounds", 10))
             for s in drift_specs[dk]]
    if a.dataset == "drugbank":
        # Fingerprint column 0 is a bit, not a degree, so degree-based node selection has no meaning.
        for e in drift:
            e.node_selection = "random"
            # "severe" means every feature, as on ogbl-ddi's 50 structural features.
            if a.drift_level == "severe":
                e.feature_dims = list(range(num_features))
    return poison, drift


def lie_z(n: int, m: int) -> float:
    """Baruch et al. (2019): the largest shift that keeps the attackers inside the honest majority."""
    from scipy.stats import norm
    s = n // 2 + 1 - m
    if s < 1:
        raise ValueError(f"A Little Is Enough needs fewer than half the clients; got {m} of {n}")
    return float(norm.ppf((n - s) / n))


def lie_updates(params: List[np.ndarray], client_arrays: List[List[np.ndarray]], lie: List[int], n: int,
                z: Optional[float] = None) -> None:
    """Replace each attacker's parameters with old + mean - z * std of the other clients' updates,
    coordinate by coordinate (an omniscient attacker, as in the original paper)."""
    z = lie_z(n, len(lie)) if z is None else z
    others = [k for k in range(n) if k not in lie]
    for li, old in enumerate(params):
        d = np.stack([client_arrays[k][li].astype(np.float64) - old for k in others])
        mal = (old + d.mean(axis=0) - z * d.std(axis=0)).astype(old.dtype)
        for k in lie:
            client_arrays[k][li] = mal.copy()


def score_model(trainer: Any, graph: torch.Tensor, parts: List[torch.Tensor], batch_size: int,
                device: torch.device) -> List[torch.Tensor]:
    """Logits of the trainer's current model on each edge set (one encoder pass)."""
    trainer.encoder.eval()
    trainer.predictor.eval()
    with torch.no_grad():
        h = trainer.encoder(graph.to(device))
        return [score_edges(h, trainer.predictor, e, batch_size, device) for e in parts]


def safety_fields(prefix: str, pos: torch.Tensor, risk: torch.Tensor, t: float) -> Dict[str, float]:
    rec = sm.recall_by_risk(pos, risk, t)
    out = {f"{prefix}_recall_{k}": v for k, v in rec.items()}
    out[f"{prefix}_mdir_high"] = sm.mdir(pos, risk, t)
    out[f"{prefix}_risk_weighted_recall"] = sm.risk_weighted_recall(pos, risk, t)
    return out


def mean_over(reports: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in reports if key in r]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    a = parse_args()
    torch.set_num_threads(a.threads)
    seed_everything(a.seed)
    device = torch.device(a.device)
    cond_tag = a.cond if a.drift_level == "base" or a.cond in ("clean", "poison") else f"{a.cond}-{a.drift_level}"
    if a.dataset == "drugbank":
        cond_tag = f"db{a.split}-{cond_tag}"
    run_name = f"{cond_tag}_{a.method}_seed{a.seed}" + (f"_{a.tag}" if a.tag else "")
    out_dir = os.path.join(a.out, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    if a.dataset == "drugbank":
        from src.data.drugbank import load_drugbank
        data, evaluator, split, node_features, atc_mapping = load_drugbank(os.path.join(a.data, "drugbank"), a.split)
        num_nodes = int(data.num_nodes)
        graph = data.edge_index
    else:
        data, evaluator = load_dataset(root=a.data)
        split = get_edge_split(root=a.data)
        num_nodes = int(data.num_nodes)
        graph = data.edge_index
        node_features = load_features(a.data, graph, num_nodes, a.seed)
        atc_path = os.path.join(a.data, "ogbl_ddi", "atc_mapping.pt")
        atc_mapping = torch.load(atc_path) if os.path.exists(atc_path) else None
    if atc_mapping is None:
        print("WARNING: no ATC mapping; poisoning 'emerge' has no effect.", flush=True)
    has_risk = "risk" in split["test"]
    if (a.hide_high or a.fabricate_high or a.log_cch) and not has_risk:
        raise SystemExit("--hide-high, --fabricate-high and --log-cch need risk labels (DrugBank built with "
                         "scripts/prepare_drugbank.py --risk-only)")
    edge_risk = None
    if a.hide_high or a.fabricate_high:
        from src.data.drugbank import edge_risk_lookup
        edge_risk = edge_risk_lookup(os.path.join(a.data, "drugbank"))
    cfg = make_cfg(a, os.path.join(out_dir, "ckpt"))

    # Mirror main_fl.py: only methods that use the server's clean edges reserve them.
    uses_root = a.method in ROOT_METHODS
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

    poison_events, drift_events = build_events(a, int(node_features.shape[1]))
    poison_sim = PoisoningSimulator(initial_datasets=datasets, train_edges=train_full, atc_mapping=atc_mapping,
                                    events=poison_events, observations_per_client=a.obs_per_client,
                                    seed=a.seed, edge_risk=edge_risk) if poison_events else None
    drift_sim = DataDriftSimulator(node_features=node_features, num_clients=a.num_clients, events=drift_events,
                                   seed=a.seed) if drift_events else None
    clients = [FedDDIClient(cid=str(i), trainer=trainer_for(i, datasets[i].edges), local_epochs=a.local_epochs,
                            device=device, num_rounds=a.rounds, round_lr_decay=a.round_lr_decay, c_id_int=i, poisoning_simulator=poison_sim,
                            data_drift_simulator=drift_sim) for i in range(a.num_clients)]
    special = sorted({e.client_id for e in poison_events} | {e.client_id for e in drift_events})
    n_attackers = len({e.client_id for e in poison_events} | set(a.lie_clients))
    byz_c = a.byz_c if a.byz_c is not None else (n_attackers or 2)
    cch_anchor = None
    if a.log_cch:
        # A fixed clean anchor for comparing defenses: validation high-risk positives and negatives.
        g = torch.Generator().manual_seed(a.seed + 4242)
        vp = split["valid"]["edge"][split["valid"]["risk"] == 2]
        vp = vp[torch.randperm(vp.size(0), generator=g)[:a.cch_anchor]]
        vn = split["valid"]["edge_neg"][torch.randperm(split["valid"]["edge_neg"].size(0), generator=g)[:a.cch_anchor]]
        cch_anchor = (vp, vn)

    eval_trainer = trainer_for(-2, client_pool[:1000])
    eval_client = FedDDIClient(cid="-2", trainer=eval_trainer, local_epochs=1, device=device)
    root_trainer = root_client = block_map = None
    hp = CAGAHyperparams(probe_size=a.probe_size, gate_blocks=a.method != "caga_noblock", tol_z=a.caga_tol_z,
                         tol_rel=a.caga_tol_rel, warmup_rounds=a.caga_warmup, class_aware=a.caga_class_aware,
                         reputation=a.caga_reputation, rep_beta=a.caga_rep_beta, rep_threshold=a.caga_rep_threshold)
    caga_state: Dict[str, Any] = {}
    # A client's share of each drug class, from its own observation mixture (coarse metadata the
    # server is assumed to know). The probe sample's class is its first drug's ATC class, the same
    # rule client sampling uses.
    mixtures = [dict(ds.class_mixture or {}) for ds in datasets]
    if uses_root:
        root_trainer = trainer_for(-1, root_edges)
        root_client = FedDDIClient(cid="-1", trainer=root_trainer, local_epochs=a.local_epochs, device=device,
                                   num_rounds=a.rounds, round_lr_decay=a.round_lr_decay)
        block_map = compute_block_index_map(root_trainer.encoder, root_trainer.predictor)
    if a.method in ("lfr", "err", "zeno") and byz_c >= a.num_clients - 1:
        raise SystemExit(f"--byz-c {byz_c} leaves too few clients")

    params = eval_client.get_parameters({})
    rows: List[Dict[str, Any]] = []
    csv_path = os.path.join(out_dir, "rounds.csv")
    t_start = time.time()
    for rnd in range(1, a.rounds + 1):
        t0 = time.time()
        fits = [c.fit(params, {"server_round": rnd}) for c in clients]
        client_arrays = [list(f[0]) for f in fits]
        if a.lie_clients and rnd >= a.attack_start:
            lie_updates(params, client_arrays, a.lie_clients, a.num_clients, a.lie_z)
        weights = [float(f[1]) for f in fits]
        row: Dict[str, Any] = {"round": rnd, "train_loss": float(np.mean([f[2]["loss"] for f in fits])),
                               "client_losses": json.dumps([round(float(f[2]["loss"]), 4) for f in fits])}

        server_arrays = None
        if a.method in ("fltrust", "fedavg_root"):
            # The server trains on a fresh client-sized sample of root edges, taking as many optimizer
            # steps as a client (FLTrust rescales client updates to this update's norm).
            g = torch.Generator().manual_seed(a.seed * 100_003 + rnd)
            pick = torch.randint(0, root_edges.size(0), (a.root_obs or a.obs_per_client,), generator=g)
            root_trainer.split["train"]["edge"] = root_edges[pick]
            server_arrays = root_client.fit(params, {"server_round": rnd})[0]
        loss_fn = probe = sw = None
        if a.method in PROBE_METHODS:
            probe = sample_probe_edges(root_pos_edges=root_trainer.split["train"]["edge"], num_nodes=num_nodes,
                                       m=hp.probe_size, seed=rnd, device=device,
                                       exclude_edge_index=root_trainer.edge_index)
            loss_fn = make_probe_loss_fn(root_trainer, block_map, probe)
            if a.caga_class_aware and atc_mapping is not None:
                cls = atc_mapping[probe[: probe.size(0) // 2, 0].cpu()].numpy()
                sw = [np.array([m.get(int(k), 0.0) for k in cls]) for m in mixtures]

        def aggregate(idx: List[int], state: Dict[str, Any]):
            """The method's aggregate of clients `idx`: (new arrays, row fields, excluded flag per client)."""
            arrs, w = [client_arrays[i] for i in idx], [weights[i] for i in idx]
            fields: Dict[str, Any] = {}
            excluded = [False] * len(idx)
            if a.method == "fedavg_root":
                # Control for CAGA's extra data: the server's root update joins plain averaging as one more client.
                arrays_all, w_all = arrs + [server_arrays], w + [float(np.mean(w))]
                tot = sum(w_all)
                new = [sum(x * y[k] for x, y in zip(w_all, arrays_all)) / tot for k in range(len(params))]
            elif a.method == "fedavg":
                tot = sum(w)
                new = [sum(x * y[k] for x, y in zip(w, arrs)) / tot for k in range(len(params))]
            elif a.method == "normclip":
                new, r = clip_and_average(params, arrs, w, a.clip_multiplier, rnd)
                fields["clipped"] = json.dumps(r.clipped_mask)
                excluded = list(r.clipped_mask)
            elif a.method == "fltrust":
                new, r = fltrust_aggregate(params, arrs, server_arrays, [str(i) for i in idx], rnd)
                fields["fltrust_trust"] = json.dumps([round(t, 3) for t in r.trust])
                excluded = [t <= 0 for t in r.trust]
            elif a.method in ("lfr", "zeno"):
                mean_loss = lambda x: float(loss_fn(x).mean())  # noqa: E731
                if a.method == "lfr":
                    new, r = lfr_aggregate(arrs, w, mean_loss, byz_c, rnd)
                else:
                    new, r = zeno_aggregate(params, arrs, w, mean_loss, byz_c, a.zeno_rho, rnd)
                fields[f"{a.method}_score"] = json.dumps([round(v, 5) for v in r.scores])
                excluded = r.excluded
            elif a.method == "err":
                n_pos = probe.size(0) // 2
                err_fn = lambda x: probe_auroc_error(  # noqa: E731
                    score_arrays_on_probe(root_trainer, block_map, x, probe), n_pos)
                new, r = err_aggregate(arrs, w, err_fn, byz_c, rnd)
                fields["err_score"] = json.dumps([round(v, 5) for v in r.scores])
                excluded = r.excluded
            elif a.method == "krum":
                new, r = krum_aggregate(params, arrs, byz_c, rnd)
                excluded = r.excluded
            elif a.method == "median":
                new, _ = median_aggregate(arrs, rnd)
            elif a.method == "trimmed":
                new, _ = trimmed_mean_aggregate(arrs, a.trim, rnd)
            else:
                new, r = caga_round(params, arrs, [str(i) for i in idx], w, block_map, loss_fn, hp, rnd,
                                    sample_weights=[sw[i] for i in idx] if sw is not None else None, state=state)
                fields.update(r.as_row_fields())
                excluded = [c.verdict != "accepted" for c in r.clients]
            return [n.astype(p.dtype, copy=False) for n, p in zip(new, params)], fields, excluded

        everyone = list(range(a.num_clients))
        if a.log_cch:
            loo_state = copy.deepcopy(caga_state)  # leave-one-out runs must not advance CAGA's reputation
        new_params, fields, excluded = aggregate(everyone, caga_state)
        row.update(fields)
        row["excluded"] = json.dumps(excluded)
        if a.log_cch:
            # CCH_i: harm of including client i, judged on the clean anchor at the deployed model's
            # 95%-specificity threshold (computed on the anchor's negatives).
            vp, vn = cch_anchor
            eval_client.set_parameters(new_params)
            pos_w, neg_w = score_model(eval_trainer, graph, [vp, vn], a.batch_size, device)
            t_anchor = sm.threshold_at_specificity(neg_w, 0.95)
            ones = torch.full((vp.size(0),), 2)
            cch = []
            for i in everyone:
                loo, _, _ = aggregate([k for k in everyone if k != i], copy.deepcopy(loo_state))
                eval_client.set_parameters(loo)
                pos_wo, neg_wo = score_model(eval_trainer, graph, [vp, vn], a.batch_size, device)
                cch.append(round(sm.cch(pos_w, pos_wo, ones, neg_w, neg_wo, t_anchor), 2))
            row["cch"] = json.dumps(cch)
            row["cch_anchor_threshold"] = t_anchor
        params = new_params

        eval_client.set_parameters(params)
        tp, tn, vp_s, vn_s = score_model(eval_trainer, graph, [split["test"]["edge"], split["test"]["edge_neg"],
                                                               split["valid"]["edge"], split["valid"]["edge_neg"]],
                                         a.batch_size, device)
        test = {k.replace("hits@", "Hits@"): float(v)
                for k, v in evaluator.eval({"y_pred_pos": tp, "y_pred_neg": tn}).items()}
        val = {k.replace("hits@", "Hits@"): float(v)
               for k, v in evaluator.eval({"y_pred_pos": vp_s, "y_pred_neg": vn_s}).items()}
        row["test_hits20"], row["val_hits20"] = test["Hits@20"], val["Hits@20"]
        for k in ("auroc", "aupr"):
            if k in test:
                row[f"test_{k}"], row[f"val_{k}"] = test[k], val[k]
        if has_risk:
            # Decision threshold at 95% specificity on the clean validation negatives, this round's model.
            t = sm.threshold_at_specificity(vn_s, 0.95)
            row["threshold"] = t
            row.update(safety_fields("test", tp, split["test"]["risk"], t))
            row.update(safety_fields("val", vp_s, split["valid"]["risk"], t))
        reports = [clients[i].evaluate(params, {"server_round": rnd})[2] for i in special]
        for key, col in (("drift_region_Hits@20", "drift_region_hits20"),
                         ("non_drift_region_Hits@20", "non_drift_region_hits20"),
                         ("poison_attack_mean_prob", "poison_attack_mean_prob"),
                         ("poison_fabricated_mean_prob", "poison_fabricated_mean_prob"),
                         ("poison_suppressed_mean_prob", "poison_suppressed_mean_prob")):
            v = mean_over(reports, key)
            if v is not None:
                row[col] = v
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        extra = f" auroc={row['test_auroc']:.4f}" if "test_auroc" in row else ""
        extra += f" mdir={row['test_mdir_high']:.4f}" if "test_mdir_high" in row else ""
        print(f"[{run_name}] round {rnd}/{a.rounds} test={row['test_hits20']:.4f} val={row['val_hits20']:.4f}"
              f"{extra} loss={row['train_loss']:.4f} {row['seconds']}s", flush=True)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "round", k))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"[{run_name}] done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
