"""OGB dataset loading and split utilities for ogbl-ddi"""

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


def _find_torch_load_calls(src: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, inner_args) for every torch.load(...) call"""
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
    """Patch OGB's dataset_pyg.py so every torch.load uses weights_only=False"""
    try:
        import ogb.linkproppred.dataset_pyg as _ogb_mod  # type: ignore
    except ImportError:
        return

    ogb_path = inspect.getfile(_ogb_mod)

    with open(ogb_path, "r", encoding="utf-8") as f:
        src = f.read()

    original = src


    calls = _find_torch_load_calls(src)
    for start, end, inner in reversed(calls):
        if ", weights_only=False" in inner:
            clean = inner.replace(", weights_only=False", "")
            src = src[:start] + f"torch.load({clean}, weights_only=False)" + src[end:]


    calls = _find_torch_load_calls(src)
    for start, end, inner in reversed(calls):
        if "weights_only" not in inner:
            src = src[:start] + f"torch.load({inner}, weights_only=False)" + src[end:]

    if src != original:
        with open(ogb_path, "w", encoding="utf-8") as f:
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


_patch_ogb_torch_load()


def load_dataset(root: str = "./data") -> Tuple[Data, Evaluator]:
    """Download (if needed) and return the ogbl-ddi PyG Data object and the official OGB evaluator."""
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
    """Return the official OGB train / valid / test edge splits"""
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
