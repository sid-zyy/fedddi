"""DrugBank DDI graph built by scripts/prepare_drugbank.py, in the same shape as the ogbl-ddi loaders.

DrugBank data is licensed and not redistributable: the processed file lives under data/ and is never
committed. Node features are Morgan fingerprints computed from each drug's SMILES.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import to_undirected  # type: ignore


class DDIEvaluator:
    """Stands in for the OGB evaluator: Hits@K against the split's negatives, plus AUROC and AUPR.

    Each split pairs every positive with one sampled non-interacting pair, so AUROC and AUPR are the
    primary metrics here; Hits@K is kept for continuity with ogbl-ddi.
    """

    def __init__(self, ks: Tuple[int, ...] = (20, 50, 100)) -> None:
        self.ks = ks

    def eval(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos = input_dict["y_pred_pos"].float().view(-1)
        neg = input_dict["y_pred_neg"].float().view(-1)
        out: Dict[str, float] = {}
        for k in self.ks:
            kth = torch.topk(neg, min(k, neg.numel())).values[-1]
            out[f"hits@{k}"] = float((pos > kth).float().mean())
        y = np.concatenate([np.ones(pos.numel()), np.zeros(neg.numel())])
        s = torch.cat([pos, neg]).numpy()
        out["auroc"] = float(roc_auc_score(y, s))
        out["aupr"] = float(average_precision_score(y, s))
        return out


def load_drugbank(root: str = "./data/drugbank", split: str = "random"
                  ) -> Tuple[Data, DDIEvaluator, Dict[str, Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Return (graph, evaluator, edge split, node features, ATC level-1 class per drug).

    The message-passing graph is the split's training edges only, so validation and test
    interactions (and, in the cold split, the held-out drugs' interactions) never leak into it.
    Files built with risk labels also give ``split[part]["risk"]`` (0 low, 1 moderate, 2 high) per positive.
    """
    path = os.path.join(root, "drugbank_ddi.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run scripts/prepare_drugbank.py on a DrugBank XML export first.")
    blob: Dict[str, Any] = torch.load(path, weights_only=False)
    if split not in blob["splits"]:
        raise ValueError(f"unknown split {split!r}; available: {sorted(blob['splits'])}")
    sp = blob["splits"][split]
    edge_split = {k: {kk: vv.long() for kk, vv in sp[k].items()} for k in ("train", "valid", "test")}
    n = int(blob["num_nodes"])
    edge_index = to_undirected(edge_split["train"]["edge"].t().contiguous(), num_nodes=n)
    data = Data(edge_index=edge_index, num_nodes=n)
    return data, DDIEvaluator(), edge_split, blob["features"].float(), blob["atc_mapping"].long()


def edge_risk_lookup(root: str = "./data/drugbank") -> Dict[Tuple[int, int], int]:
    """Risk class (0 low, 1 moderate, 2 high) of every interacting pair, keyed by ``(min(u, v), max(u, v))``."""
    blob: Dict[str, Any] = torch.load(os.path.join(root, "drugbank_ddi.pt"), weights_only=False)
    if "edge_risk" not in blob:
        raise KeyError("no risk labels: run scripts/prepare_drugbank.py --risk-only --out " + root)
    e = blob["edges"].long()
    lo, hi = torch.minimum(e[:, 0], e[:, 1]).tolist(), torch.maximum(e[:, 0], e[:, 1]).tolist()
    return dict(zip(zip(lo, hi), blob["edge_risk"].long().tolist()))
