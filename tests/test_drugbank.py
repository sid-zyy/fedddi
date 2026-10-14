"""DrugBank loader and evaluator on a small synthetic file (no DrugBank data in tests)."""
from __future__ import annotations

import pytest
import torch

from src.data.ddi_risk import RISK_CODE, RiskMap, make_template
from src.data.drugbank import DDIEvaluator, edge_risk_lookup, load_drugbank


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


def test_template_replaces_both_names_longest_first():
    desc = "Warfarin may increase the anticoagulant activities of Warfarin sodium."
    assert make_template(desc, ("Warfarin", "Warfarin sodium")) == "#DRUG may increase the anticoagulant activities of #DRUG."


@pytest.mark.parametrize("template, typ, risk", [
    ("The risk or severity of CNS depression can be increased when #DRUG is combined with #DRUG.", "cns_depression", "high"),
    ("The risk or severity of QTc prolongation and torsade de pointes can be increased when #DRUG is combined with Salbutamol.",
     "qtc_prolongation", "high"),
    ("#DRUG may increase the anticoagulant activities of #DRUG.", "bleeding", "high"),
    ("#DRUG may decrease the anticoagulant activities of #DRUG.", "efficacy_change", "moderate"),
    ("#DRUG may decrease the hypoglycemic activities of #DRUG.", "efficacy_change", "moderate"),
    ("#DRUG may decrease the nephrotoxic activities of #DRUG.", "risk_decrease", "low"),
    ("The risk or severity of serotonin syndrome can be decreased when #DRUG is combined with #DRUG.", "risk_decrease", "low"),
    ("The metabolism of #DRUG can be decreased when combined with Rifampicin.", "pk_change", "moderate"),
    ("The therapeutic efficacy of #DRUG can be decreased when used in combination with #DRUG.", "efficacy_change", "moderate"),
    ("#DRUG may increase the antihypertensive activities of #DRUG.", "efficacy_change", "moderate"),
    ("#DRUG may increase the orthostatic hypotensive activities of #DRUG.", "hypotension", "high"),
    ("The risk or severity of adverse effects can be increased when #DRUG is combined with #DRUG.", "other", "low"),
    ("The risk or severity of hypertension can be increased when #DRUG is combined with #DRUG.", "other", "low"),
    ("Something DrugBank has never written.", "other", "low"),
])
def test_shipped_risk_map(template, typ, risk):
    assert RiskMap.load().classify(template) == (typ, RISK_CODE[risk])


def test_risk_map_first_match_wins_and_validates():
    rm = RiskMap([{"template": "exact #DRUG", "type": "a", "risk": "moderate"},
                  {"pattern": "bleed", "type": "b", "risk": "high"}])
    assert rm.classify("exact #DRUG") == ("a", 1)
    assert rm.classify("BLEEDING risk") == ("b", 2)  # case-insensitive search
    assert rm.classify("nothing") == ("other", 0)
    assert rm.types == ["a", "b", "other"] and rm.type_id("b") == 1
    with pytest.raises(ValueError):
        RiskMap([{"pattern": "x", "type": "t", "risk": "severe"}])
    with pytest.raises(ValueError):
        RiskMap([{"pattern": "x", "template": "x", "type": "t", "risk": "low"}])


def test_label_risk_aligns_with_edges_and_splits(tmp_path):
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "prepare_drugbank", os.path.join(os.path.dirname(__file__), "..", "scripts", "prepare_drugbank.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    names = ["Aspirin", "Warfarin", "Zolpidem", "Diazepam"]
    rows = [(0, 1, "Aspirin may increase the anticoagulant activities of Warfarin."),
            (2, 3, "The risk or severity of CNS depression can be increased when Zolpidem is combined with Diazepam."),
            (1, 2, "The metabolism of Zolpidem can be decreased when combined with Warfarin."),
            (3, 0, "The risk or severity of adverse effects can be increased when Diazepam is combined with Aspirin.")]
    edges = torch.tensor([[0, 1], [2, 3], [1, 2], [0, 3]])
    blob = {"num_nodes": 4, "edges": edges,
            "splits": {"random": {"train": {"edge": edges[[1, 2]]},
                                  "valid": {"edge": torch.tensor([[3, 0]]), "edge_neg": torch.tensor([[0, 2]])},
                                  "test": {"edge": edges[[0]], "edge_neg": torch.tensor([[1, 3]])}}}}
    prep.label_risk(blob, rows, names, RiskMap.load(), str(tmp_path))
    assert blob["edge_risk"].tolist() == [2, 2, 1, 0]
    sp = blob["splits"]["random"]
    assert sp["train"]["risk"].tolist() == [2, 1]
    assert sp["valid"]["risk"].tolist() == [0]  # reversed pair order still found
    assert sp["test"]["risk"].tolist() == [2]
    assert (tmp_path / "templates.csv").read_text(encoding="utf-8").startswith("template,count,type,risk")
    with pytest.raises(ValueError):
        prep.label_risk(blob, rows[::-1], names, RiskMap.load(), str(tmp_path))


def test_loader_returns_risk_and_lookup(tmp_path):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]])
    risk = torch.tensor([2, 0, 1, 2, 0])
    split = {"train": {"edge": edges[:3], "risk": risk[:3]},
             "valid": {"edge": edges[3:4], "edge_neg": torch.tensor([[0, 2]]), "risk": risk[3:4]},
             "test": {"edge": edges[4:], "edge_neg": torch.tensor([[0, 2]]), "risk": risk[4:]}}
    torch.save({"num_nodes": 4, "features": torch.eye(4), "atc_mapping": torch.tensor([0, 1, 14, 2]),
                "edges": edges, "edge_risk": risk, "splits": {"random": split}}, tmp_path / "drugbank_ddi.pt")
    _, _, sp, _, _ = load_drugbank(str(tmp_path), "random")
    assert sp["train"]["risk"].tolist() == [2, 0, 1] and sp["valid"]["risk"].tolist() == [2]
    lut = edge_risk_lookup(str(tmp_path))
    assert lut[(0, 1)] == 2 and lut[(1, 3)] == 0 and len(lut) == 5
