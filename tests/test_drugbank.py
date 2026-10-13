"""DrugBank loader and evaluator on a small synthetic file (no DrugBank data in tests)."""
from __future__ import annotations

import pytest
import torch

from src.data.drugbank import DDIEvaluator, load_drugbank


def test_evaluator_perfect_and_random_scores():
    ev = DDIEvaluator(ks=(2,))
    perfect = ev.eval({"y_pred_pos": torch.tensor([0.9, 0.8, 0.7]), "y_pred_neg": torch.tensor([0.1, 0.2, 0.3])})
    assert perfect["auroc"] == pytest.approx(1.0) and perfect["aupr"] == pytest.approx(1.0)
    assert perfect["hits@2"] == pytest.approx(1.0)
    flipped = ev.eval({"y_pred_pos": torch.tensor([0.1, 0.2]), "y_pred_neg": torch.tensor([0.8, 0.9])})
    assert flipped["auroc"] == pytest.approx(0.0) and flipped["hits@2"] == pytest.approx(0.0)


def _blob(tmp_path):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]])
    split = {"train": {"edge": edges[:3]}, "valid": {"edge": edges[3:4], "edge_neg": torch.tensor([[0, 2]])},
             "test": {"edge": edges[4:], "edge_neg": torch.tensor([[0, 2]])}}
    torch.save({"num_nodes": 4, "features": torch.eye(4), "atc_mapping": torch.tensor([0, 1, 14, 2]),
                "edges": edges, "splits": {"random": split}}, tmp_path / "drugbank_ddi.pt")


def test_loader_builds_graph_from_training_edges_only(tmp_path):
    _blob(tmp_path)
    data, ev, split, feats, atc = load_drugbank(str(tmp_path), "random")
    assert data.num_nodes == 4 and feats.shape == (4, 4) and atc.tolist() == [0, 1, 14, 2]
    pairs = {tuple(p) for p in data.edge_index.t().tolist()}
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}  # held-out edges are not in the graph
    assert split["valid"]["edge"].tolist() == [[0, 3]]


def test_loader_rejects_unknown_split_and_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_drugbank(str(tmp_path), "random")
    _blob(tmp_path)
    with pytest.raises(ValueError):
        load_drugbank(str(tmp_path), "cold")
