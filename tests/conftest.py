"""
tests/conftest.py
─────────────────
Shared fixtures for the test suite.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data  # type: ignore

NUM_NODES = 50
NUM_EDGES = 200
HIDDEN_DIM = 32


@pytest.fixture
def synthetic_graph() -> Data:
    """A small random graph for fast tests."""
    src = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    edge_index = torch.stack([src, dst], dim=0)
    data = Data(edge_index=edge_index, num_nodes=NUM_NODES)
    return data


@pytest.fixture
def edge_index(synthetic_graph: Data) -> torch.Tensor:
    assert synthetic_graph.edge_index is not None
    return synthetic_graph.edge_index


@pytest.fixture
def synthetic_split(edge_index: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    """Minimal train/valid/test split mirroring OGB structure."""
    n = edge_index.size(1)
    edges_t = edge_index.t()

    def _rand_neg(k: int) -> torch.Tensor:
        s = torch.randint(0, NUM_NODES, (k,))
        d = torch.randint(0, NUM_NODES, (k,))
        return torch.stack([s, d], dim=1)

    return {
        "train": {"edge": edges_t[: int(0.7 * n)]},
        "valid": {
            "edge": edges_t[int(0.7 * n) : int(0.85 * n)],
            "edge_neg": _rand_neg(int(0.15 * n)),
        },
        "test": {
            "edge": edges_t[int(0.85 * n) :],
            "edge_neg": _rand_neg(int(0.15 * n)),
        },
    }

