"""OGB dataset loading and split utilities for ogbl-ddi"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Tuple, cast, Any

import torch
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import to_undirected  # type: ignore

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _trusted_ogb_load():
    """Let OGB's own torch.load calls unpickle its processed dataset files.

    PyTorch >= 2.6 defaults torch.load to weights_only=True, which rejects the PyG objects OGB
    caches. These files are produced locally by OGB from the official download, so inside this
    block torch.load defaults to weights_only=False. Callers that pass weights_only explicitly
    are unaffected, and the original function is restored on exit.
    """
    original = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def load_dataset(root: str = "./data") -> Tuple[Data, Evaluator]:
    """Download (if needed) and return the ogbl-ddi PyG Data object and the official OGB evaluator."""
    logger.info("Loading ogbl-ddi dataset from '%s' ...", root)
    with _trusted_ogb_load():
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)

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
    with _trusted_ogb_load():
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
        split = cast(
            Dict[str, Dict[str, torch.Tensor]], cast(Any, dataset).get_edge_split()
        )

    logger.info(
        "Edge split — train: %d, valid: %d (neg %d), test: %d (neg %d)",
        split["train"]["edge"].size(0),
        split["valid"]["edge"].size(0),
        split["valid"]["edge_neg"].size(0),
        split["test"]["edge"].size(0),
        split["test"]["edge_neg"].size(0),
    )
    return split
