"""Build a second labelled DDI dataset from DDInter, in the same format as the DrugBank build.

DDInter (CC BY-NC-SA 4.0) grades each interaction Major / Moderate / Minor / Unknown. We keep pairs whose two
drugs match a drug in our DrugBank build by normalised name, drop "Unknown" grades, and map Major -> high (2),
Moderate -> moderate (1), Minor -> low (0). Node features and ATC classes are taken from the DrugBank build
(Morgan fingerprints), so the model input is identical; the interactions and their severity come from DDInter.
Because the features are DrugBank-derived, the output must stay private like the DrugBank build.

Writes <out>/ddinter_ddi.pt (loaded with --db-dir) and <out>/drugs.csv.

Usage: python scripts/prepare_ddinter.py [--ddinter data/ddinter] [--drugbank data/drugbank] [--out data/ddinter_graph]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_drugbank import sample_negatives  # noqa: E402

GRADE = {"Minor": 0, "Moderate": 1, "Major": 2}


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ddinter", default="data/ddinter")
    p.add_argument("--drugbank", default="data/drugbank")
    p.add_argument("--out", default="data/ddinter_graph")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dd = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.ddinter, "ddinter_downloads_code_*.csv")))],
                   ignore_index=True)
    db = pd.read_csv(os.path.join(a.drugbank, "drugs.csv"))
    blob_db = torch.load(os.path.join(a.drugbank, "drugbank_ddi.pt"), weights_only=False)
    index = {norm(n): i for i, n in enumerate(db["name"].fillna(""))}

    dd = dd[dd.Level.isin(GRADE)].copy()
    dd["x"], dd["y"] = dd.Drug_A.map(norm).map(index), dd.Drug_B.map(norm).map(index)
    dd = dd.dropna(subset=["x", "y"])
    dd = dd[dd.x != dd.y]
    dd["lo"], dd["hi"] = dd[["x", "y"]].min(1).astype(int), dd[["x", "y"]].max(1).astype(int)
    dd["r"] = dd.Level.map(GRADE)
    pairs = dd.groupby(["lo", "hi"]).r.max().reset_index()  # most severe grade if listed more than once

    used = np.unique(pairs[["lo", "hi"]].values)
    remap = -np.ones(len(db), dtype=np.int64)
    remap[used] = np.arange(len(used))
    n = len(used)
    edges = np.stack([remap[pairs.lo.values], remap[pairs.hi.values]], 1).astype(np.int64)
    risk = pairs.r.values.astype(np.int64)
    pos = set(map(tuple, edges.tolist()))
    print(f"{n} drugs, {len(edges)} graded interactions, density {2 * len(edges) / (n * (n - 1)):.3f}; "
          f"risk shares low/moderate/high {[round(float((risk == k).mean()), 3) for k in range(3)]}")

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(edges))
    nv, nt = int(a.val_frac * len(edges)), int(a.test_frac * len(edges))
    allowed = np.ones(n, dtype=bool)
    parts = {"train": perm[nv + nt:], "valid": perm[:nv], "test": perm[nv:nv + nt]}
    split = {k: {"edge": torch.from_numpy(edges[idx]), "risk": torch.from_numpy(risk[idx])} for k, idx in parts.items()}
    for k, cnt in (("valid", nv), ("test", nt)):
        split[k]["edge_neg"] = torch.from_numpy(sample_negatives(n, pos, cnt, rng, allowed))

    out = {"num_nodes": n, "features": blob_db["features"][torch.from_numpy(used)].clone(),
           "atc_mapping": blob_db["atc_mapping"][torch.from_numpy(used)].clone(),
           "edges": torch.from_numpy(edges), "edge_risk": torch.from_numpy(risk), "splits": {"random": split},
           "meta": {"source": "DDInter (CC BY-NC-SA 4.0) interactions and grades; DrugBank fingerprints and ATC",
                    "risk_names": ["low", "moderate", "high"], "seed": a.seed}}
    os.makedirs(a.out, exist_ok=True)
    torch.save(out, os.path.join(a.out, "ddinter_ddi.pt"))
    db.iloc[used].assign(index=np.arange(n)).to_csv(os.path.join(a.out, "drugs.csv"), index=False)
    print("wrote", os.path.join(a.out, "ddinter_ddi.pt"),
          {k: int(v["edge"].shape[0]) for k, v in split.items()},
          "train high-risk share", round(float((split["train"]["risk"] == 2).float().mean()), 3))


if __name__ == "__main__":
    main()
