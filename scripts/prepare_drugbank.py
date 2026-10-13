"""Build a drug-drug interaction graph from a licensed DrugBank XML export.

DrugBank data may not be redistributed: the XML and everything written here stay under data/, which
is excluded from Git. Streams the XML once and keeps small-molecule drugs that have a SMILES string.

Writes to --out:
  drugs.csv              drugbank_id, name, groups, atc_level1, smiles, degree
  interactions.csv.gz    one row per unordered pair: a, b (node indices), description
  drugbank_ddi.pt        features (Morgan fingerprints), atc_mapping, edges, random and cold-drug splits

Example:
    python scripts/prepare_drugbank.py --xml "path/to/full database.xml" --out data/drugbank
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

NS = "{http://www.drugbank.ca}"
# Same class order as scripts/prepare_atc_mapping.py; index 14 = unknown.
ATC_CODES = ["A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xml", required=True)
    p.add_argument("--out", default="data/drugbank")
    p.add_argument("--fp-bits", type=int, default=1024)
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--approved-only", action="store_true", help="Keep only drugs in the 'approved' group.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    return p.parse_args()


def parse_xml(path: str) -> Tuple[List[dict], Dict[Tuple[str, str], str]]:
    """Top-level drug records and their interactions keyed by an unordered id pair."""
    from lxml import etree

    drugs: List[dict] = []
    pairs: Dict[Tuple[str, str], str] = {}
    root_tag = NS + "drugbank"
    for _, el in etree.iterparse(path, events=("end",), tag=NS + "drug", huge_tree=True):
        if el.getparent() is None or el.getparent().tag != root_tag:
            continue  # <drug> entries nested inside pathways, not drug records
        pid = next((d.text for d in el.findall(NS + "drugbank-id") if d.get("primary") == "true"), None)
        smiles = None
        for prop in el.iterfind(f"{NS}calculated-properties/{NS}property"):
            if prop.findtext(NS + "kind") == "SMILES":
                smiles = prop.findtext(NS + "value")
                break
        atc = [a.get("code") for a in el.iterfind(f"{NS}atc-codes/{NS}atc-code") if a.get("code")]
        groups = [g.text for g in el.iterfind(f"{NS}groups/{NS}group") if g.text]
        partners = []
        for it in el.iterfind(f"{NS}drug-interactions/{NS}drug-interaction"):
            other = it.findtext(NS + "drugbank-id")
            if other:
                partners.append((other, (it.findtext(NS + "description") or "").strip()))
        drugs.append(dict(id=pid, name=el.findtext(NS + "name") or "", type=el.get("type"), smiles=smiles,
                          atc=atc, groups=groups))
        for other, desc in partners:
            key = (pid, other) if pid < other else (other, pid)
            pairs.setdefault(key, desc)
        # Free the finished record and the siblings already processed.
        el.clear()
        parent = el.getparent()
        while el.getprevious() is not None:
            del parent[0]
    return drugs, pairs


def fingerprints(smiles: List[str], bits: int, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    feats = np.zeros((len(smiles), bits), dtype=np.uint8)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is not None:
            feats[i] = gen.GetFingerprintAsNumPy(mol)
            ok[i] = True
    return feats, ok


def sample_negatives(n: int, pos: Set[Tuple[int, int]], count: int, rng: np.random.Generator,
                     allowed: np.ndarray, need_one: np.ndarray | None = None) -> np.ndarray:
    """Unordered non-edges among `allowed` nodes; with `need_one`, at least one end must be in it."""
    nodes = np.flatnonzero(allowed)
    special = np.flatnonzero(need_one) if need_one is not None else None
    out: Set[Tuple[int, int]] = set()
    while len(out) < count:
        k = (count - len(out)) * 2
        u = rng.choice(special if special is not None else nodes, k)
        v = rng.choice(nodes, k)
        for a, b in zip(u.tolist(), v.tolist()):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in pos and key not in out:
                out.add(key)
                if len(out) == count:
                    break
    return np.array(sorted(out), dtype=np.int64)


def main() -> None:
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("Parsing", a.xml, flush=True)
    drugs, pairs = parse_xml(a.xml)
    print(f"{len(drugs)} drug records, {len(pairs)} unordered interaction pairs", flush=True)

    keep = [d for d in drugs if d["type"] == "small molecule" and d["smiles"]
            and (not a.approved_only or "approved" in d["groups"])]
    feats, ok = fingerprints([d["smiles"] for d in keep], a.fp_bits, a.fp_radius)
    keep = [d for d, good in zip(keep, ok) if good]
    feats = feats[ok]
    index = {d["id"]: i for i, d in enumerate(keep)}
    rows = [(index[x], index[y], desc) for (x, y), desc in pairs.items() if x in index and y in index]

    # Drop drugs left without interactions inside the kept set, then re-index.
    deg = np.zeros(len(keep), dtype=np.int64)
    for x, y, _ in rows:
        deg[x] += 1
        deg[y] += 1
    live = np.flatnonzero(deg > 0)
    remap = -np.ones(len(keep), dtype=np.int64)
    remap[live] = np.arange(len(live))
    keep = [keep[i] for i in live]
    feats, deg = feats[live], deg[live]
    rows = [(int(remap[x]), int(remap[y]), desc) for x, y, desc in rows]
    n = len(keep)
    edges = np.array([(x, y) if x < y else (y, x) for x, y, _ in rows], dtype=np.int64)
    pos = set(map(tuple, edges.tolist()))
    print(f"kept {n} small-molecule drugs with SMILES and interactions; {len(edges)} interaction pairs; "
          f"density {2 * len(edges) / (n * (n - 1)):.3f}", flush=True)

    atc = np.array([ATC_CODES.index(d["atc"][0][0]) if d["atc"] and d["atc"][0][0] in ATC_CODES else len(ATC_CODES)
                    for d in keep], dtype=np.int64)

    rng = np.random.default_rng(a.seed)
    # Random split over pairs.
    perm = rng.permutation(len(edges))
    nv, nt = int(a.val_frac * len(edges)), int(a.test_frac * len(edges))
    allowed = np.ones(n, dtype=bool)
    rand = {"train": {"edge": edges[perm[nv + nt:]]},
            "valid": {"edge": edges[perm[:nv]], "edge_neg": sample_negatives(n, pos, nv, rng, allowed)},
            "test": {"edge": edges[perm[nv:nv + nt]], "edge_neg": sample_negatives(n, pos, nt, rng, allowed)}}
    # Cold-drug split: validation and test drugs are unseen in training.
    dperm = rng.permutation(n)
    dv, dt = int(a.val_frac * n), int(a.test_frac * n)
    role = np.zeros(n, dtype=np.int64)  # 0 train, 1 valid, 2 test
    role[dperm[:dv]], role[dperm[dv:dv + dt]] = 1, 2
    ra, rb = role[edges[:, 0]], role[edges[:, 1]]
    tr = (ra == 0) & (rb == 0)
    va = ((ra == 1) | (rb == 1)) & (ra != 2) & (rb != 2)
    te = (ra == 2) | (rb == 2)
    cold = {"train": {"edge": edges[tr]},
            "valid": {"edge": edges[va], "edge_neg": sample_negatives(n, pos, int(va.sum()), rng, role != 2, role == 1)},
            "test": {"edge": edges[te], "edge_neg": sample_negatives(n, pos, int(te.sum()), rng, allowed, role == 2)},
            "drug_role": torch.from_numpy(role)}
    to_t = lambda s: {k: ({kk: torch.from_numpy(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                      for k, v in s.items()}

    torch.save({"num_nodes": n, "features": torch.from_numpy(feats.astype(np.float32)),
                "atc_mapping": torch.from_numpy(atc), "edges": torch.from_numpy(edges),
                "splits": {"random": to_t(rand), "cold": to_t(cold)},
                "meta": {"source": os.path.basename(a.xml), "fp_bits": a.fp_bits, "fp_radius": a.fp_radius,
                         "approved_only": a.approved_only, "seed": a.seed}},
               os.path.join(a.out, "drugbank_ddi.pt"))
    with open(os.path.join(a.out, "drugs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "drugbank_id", "name", "groups", "atc_level1", "smiles", "degree"])
        for i, d in enumerate(keep):
            w.writerow([i, d["id"], d["name"], "|".join(d["groups"]),
                        ATC_CODES[atc[i]] if atc[i] < len(ATC_CODES) else "", d["smiles"], int(deg[i])])
    with gzip.open(os.path.join(a.out, "interactions.csv.gz"), "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "description"])
        w.writerows(rows)
    sizes = {k: (len(s["train"]["edge"]), len(s["valid"]["edge"]), len(s["test"]["edge"]))
             for k, s in (("random", rand), ("cold", cold))}
    print("split sizes (train, valid, test):", sizes)
    print("ATC level-1 known for", int((atc < len(ATC_CODES)).sum()), "of", n, "drugs")
    print("Saved to", a.out)


if __name__ == "__main__":
    main()
