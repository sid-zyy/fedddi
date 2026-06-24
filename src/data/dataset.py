"""
src/data/dataset.py
───────────────────
OGB dataset loading and split utilities for ogbl-ddi.

The ogbl-ddi graph:
  • 4,267 nodes  (drug entities, no input node features)
  • 1,334,889 edges
  • Task: predict drug–drug interactions (Hits@20)

Compatibility fix (PyTorch >= 2.6)
───────────────────────────────────
PyTorch 2.6+ changed torch.load default to weights_only=True.
OGB's dataset_pyg.py uses torch.load without that flag, causing
UnpicklingError. _patch_ogb_torch_load() patches OGB in-place using a
paren-aware scanner (not regex) so weights_only=False is correctly placed
AFTER the closing paren of any nested call like osp.join(...).

Handles all three possible states of the OGB file:
  1. Unpatched   — adds weights_only=False
  2. Broken patch — weights_only was put INSIDE osp.join args, fixes it
  3. Correct      — leaves it unchanged

Runs automatically on import. Safe and idempotent.
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
from typing import Dict, List, Tuple, cast, Any

import torch
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import to_undirected  # type: ignore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch >= 2.6 compatibility patch
# ─────────────────────────────────────────────────────────────────────────────


def _find_torch_load_calls(src: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, inner_args) for every torch.load(...) call.

    Uses a character-by-character paren counter so nested calls like
    torch.load(osp.join(a, b)) are handled correctly — the regex approach
    fails here because [^)]+ stops at the first ) it sees.
    """
    results: List[Tuple[int, int, str]] = []
    i = 0
    needle = "torch.load("
    while i < len(src):
        idx = src.find(needle, i)
        if idx == -1:
            break
        depth = 0
        j = idx + len(needle)
        start_inner = j
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                if depth == 0:
                    results.append((idx, j + 1, src[start_inner:j]))
                    break
                depth -= 1
            j += 1
        i = idx + 1
    return results


def _patch_ogb_torch_load() -> None:
    """Patch OGB's dataset_pyg.py so every torch.load uses weights_only=False.

    Two-pass approach:
      Pass 1 — fix broken patches: remove weights_only from inside nested args,
               place it correctly after torch.load's own closing paren.
      Pass 2 — add weights_only=False to any call that still lacks it.
    """
    try:
        import ogb.linkproppred.dataset_pyg as _ogb_mod  # type: ignore
    except ImportError:
        return  # will fail later with a clear ImportError

    ogb_path = inspect.getfile(_ogb_mod)

    with open(ogb_path, "r") as f:
        src = f.read()

    original = src

    # ── Pass 1: fix broken state (weights_only mistakenly inside nested args)
    calls = _find_torch_load_calls(src)
    for start, end, inner in reversed(calls):
        if ", weights_only=False" in inner:
            clean = inner.replace(", weights_only=False", "")
            src = src[:start] + f"torch.load({clean}, weights_only=False)" + src[end:]

    # ── Pass 2: add weights_only=False where still missing
    calls = _find_torch_load_calls(src)
    for start, end, inner in reversed(calls):
        if "weights_only" not in inner:
            src = src[:start] + f"torch.load({inner}, weights_only=False)" + src[end:]

    if src != original:
        with open(ogb_path, "w") as f:
            f.write(src)
        count = src.count("weights_only=False")
        logger.debug(
            "OGB patched — %d torch.load call(s) updated in %s", count, ogb_path
        )


def _clear_ogb_cache(root: str) -> None:
    """Delete OGB processed cache so it regenerates with the patched loader."""
    processed = os.path.join(root, "ddi", "processed")
    if os.path.isdir(processed):
        shutil.rmtree(processed)
        logger.debug("Cleared OGB cache: %s", processed)


# Apply patch immediately on import
_patch_ogb_torch_load()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(root: str = "./data") -> Tuple[Data, Evaluator]:
    """Download (if needed) and return the ogbl-ddi PyG Data object and
    the official OGB evaluator.
    """
    logger.info("Loading ogbl-ddi dataset from '%s' ...", root)
    _patch_ogb_torch_load()

    try:
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
    except Exception as e:
        if "weights_only" in str(e) or "Unpickling" in type(e).__name__:
            logger.warning("torch.load failed — clearing cache and retrying ...")
            _clear_ogb_cache(root)
            _patch_ogb_torch_load()
            dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
        else:
            raise

    data = cast(Data, dataset[0])
    edge_index = cast(torch.Tensor, data.edge_index)
    data.edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    logger.info(
        "Dataset loaded — nodes: %d, edges (undirected): %d",
        data.num_nodes,
        data.edge_index.size(1) // 2,
    )

    evaluator = Evaluator(name="ogbl-ddi")
    return data, evaluator


def get_edge_split(root: str = "./data") -> Dict[str, Dict[str, torch.Tensor]]:
    """Return the official OGB train / valid / test edge splits.

    Structure:
        split["train"]["edge"]        — [N_train, 2] positive edges
        split["valid"]["edge"]        — [N_valid, 2] positive edges
        split["valid"]["edge_neg"]    — [N_valid, 2] negative edges
        split["test"]["edge"]         — [N_test,  2] positive edges
        split["test"]["edge_neg"]     — [N_test,  2] negative edges
    """
    _patch_ogb_torch_load()

    try:
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
        split = cast(
            Dict[str, Dict[str, torch.Tensor]], cast(Any, dataset).get_edge_split()
        )
    except Exception as e:
        if "weights_only" in str(e) or "Unpickling" in type(e).__name__:
            logger.warning("torch.load failed — clearing cache and retrying ...")
            _clear_ogb_cache(root)
            _patch_ogb_torch_load()
            dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
            split = cast(
                Dict[str, Dict[str, torch.Tensor]], cast(Any, dataset).get_edge_split()
            )
        else:
            raise

    logger.info(
        "Edge split — train: %d, valid: %d (neg %d), test: %d (neg %d)",
        split["train"]["edge"].size(0),
        split["valid"]["edge"].size(0),
        split["valid"]["edge_neg"].size(0),
        split["test"]["edge"].size(0),
        split["test"]["edge_neg"].size(0),
    )
    return split
