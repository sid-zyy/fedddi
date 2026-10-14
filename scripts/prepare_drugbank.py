"""Build a drug-drug interaction graph from a licensed DrugBank XML export.

DrugBank data may not be redistributed: the XML and everything written here stay under data/, which
is excluded from Git. Streams the XML once and keeps small-molecule drugs that have a SMILES string.

Writes to --out:
  drugs.csv              drugbank_id, name, groups, atc_level1, smiles, degree
  interactions.csv.gz    one row per unordered pair: a, b (node indices), description
  drugbank_ddi.pt        features (Morgan fingerprints), atc_mapping, edges, random and cold-drug splits,
                         edge_type / edge_risk per pair and a "risk" tensor for every split's positives
  templates.csv          template, count, type, risk (the risk mapping, for the paper's appendix)

Risk classes come from conf/ddi_risk_map.yaml (src/data/ddi_risk.py). --risk-only relabels an existing
build from its drugs.csv and interactions.csv.gz without re-parsing the XML, keeping the splits unchanged.

Example:
    python scripts/prepare_drugbank.py --xml "path/to/full database.xml" --out data/drugbank
    python scripts/prepare_drugbank.py --risk-only --out data/drugbank
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.data.ddi_risk import DEFAULT_MAP, RISK_NAMES, RiskMap, make_template  # noqa: E402

NS = "{http://www.drugbank.ca}"
# Same class order as scripts/prepare_atc_mapping.py; index 14 = unknown.
ATC_CODES = ["A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xml", help="DrugBank XML export (not needed with --risk-only).")
    p.add_argument("--risk-only", action="store_true",
                   help="Only (re)compute risk labels for the existing build in --out.")
    p.add_argument("--risk-map", default=DEFAULT_MAP)
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


def pair_keys(edges: np.ndarray, n: int) -> np.ndarray:
    lo, hi = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
    return lo.astype(np.int64) * n + hi


def label_risk(blob: dict, rows: Sequence[Tuple[int, int, str]], names: Sequence[str], risk_map: RiskMap,
               out_dir: str) -> None:
    """Add edge_type, edge_risk and per-split positive risk to `blob`; write templates.csv."""
    edges = blob["edges"].numpy()
    if len(rows) != len(edges) or any((min(x, y), max(x, y)) != (int(e[0]), int(e[1]))
                                      for (x, y, _), e in zip(rows, edges)):
        raise ValueError("interaction rows are not aligned with the saved edges; rebuild from the XML")
    templates = [make_template(desc, (names[x], names[y])) for x, y, desc in rows]
    labels = [risk_map.classify(t) for t in templates]
    edge_type = np.array([risk_map.type_id(t) for t, _ in labels], dtype=np.int64)
    edge_risk = np.array([r for _, r in labels], dtype=np.int64)
    n = int(blob["num_nodes"])
    keys = pair_keys(edges, n)
    order = np.argsort(keys)
    for split in blob["splits"].values():
        for part in ("train", "valid", "test"):
            q = pair_keys(split[part]["edge"].numpy(), n)
            idx = order[np.searchsorted(keys, q, sorter=order)]
            if not np.array_equal(keys[idx], q):
                raise ValueError("a split edge is missing from the edge list")
            split[part]["risk"] = torch.from_numpy(edge_risk[idx])
    blob["edge_type"], blob["edge_risk"] = torch.from_numpy(edge_type), torch.from_numpy(edge_risk)
    blob["meta"] = {**blob.get("meta", {}), "edge_types": risk_map.types, "risk_names": list(RISK_NAMES),
                    "risk_map": os.path.basename(risk_map.path)}
    counts = Counter(templates)
    with open(os.path.join(out_dir, "templates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["template", "count", "type", "risk"])
        for t, c in counts.most_common():
            typ, r = risk_map.classify(t)
            w.writerow([t, c, typ, RISK_NAMES[r]])
    print("risk classes:", {RISK_NAMES[r]: f"{int((edge_risk == r).sum())} ({(edge_risk == r).mean():.1%})"
                            for r in range(len(RISK_NAMES))})
    by_type = Counter(risk_map.types[i] for i in edge_type.tolist())
    print("types:", dict(by_type.most_common()))
    for name, split in blob["splits"].items():
        print(f"{name} split high-risk share:",
              {p: f"{float((split[p]['risk'] == 2).float().mean()):.3f}" for p in ("train", "valid", "test")})


def relabel_existing(out_dir: str, risk_map: RiskMap) -> None:
    import pandas as pd

    path = os.path.join(out_dir, "drugbank_ddi.pt")
    blob = torch.load(path, weights_only=False)
    names = pd.read_csv(os.path.join(out_dir, "drugs.csv"))["name"].fillna("").astype(str).tolist()
    it = pd.read_csv(os.path.join(out_dir, "interactions.csv.gz"))
    rows = list(zip(it["a"].astype(int).tolist(), it["b"].astype(int).tolist(),
                    it["description"].fillna("").astype(str).tolist()))
    label_risk(blob, rows, names, risk_map, out_dir)
    torch.save(blob, path)
    print("Relabelled", path)


def main() -> None:
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    risk_map = RiskMap.load(a.risk_map)
    if a.risk_only:
        relabel_existing(a.out, risk_map)
        return
    if not a.xml:
        raise SystemExit("--xml is required unless --risk-only is given")
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

    blob = {"num_nodes": n, "features": torch.from_numpy(feats.astype(np.float32)),
            "atc_mapping": torch.from_numpy(atc), "edges": torch.from_numpy(edges),
            "splits": {"random": to_t(rand), "cold": to_t(cold)},
            "meta": {"source": os.path.basename(a.xml), "fp_bits": a.fp_bits, "fp_radius": a.fp_radius,
                     "approved_only": a.approved_only, "seed": a.seed}}
    label_risk(blob, rows, [d["name"] for d in keep], risk_map, a.out)
    torch.save(blob, os.path.join(a.out, "drugbank_ddi.pt"))
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
