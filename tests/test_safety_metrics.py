"""Safety metrics (MDIR, recall by risk, CCH, DRR) on hand-computed toy scores."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.training.safety_metrics import (
    cch,
    drr,
    mdir,
    recall_by_risk,
    risk_weighted_recall,
    threshold_at_specificity,
)


def test_threshold_leaves_spec_share_of_negatives_below():
    neg = np.arange(100, dtype=float)  # 0..99
    t = threshold_at_specificity(neg, 0.95)
    assert t == 95.0 and (neg < t).mean() == 0.95
    assert threshold_at_specificity(torch.tensor([3.0, 1.0, 2.0]), 0.5) == 3.0  # at 2.0 only 1/3 is below
    assert threshold_at_specificity([1.0, 2.0], 1.0) == 2.0  # clamped to the largest score
    with pytest.raises(ValueError):
        threshold_at_specificity([], 0.95)


def test_mdir_and_recall_by_risk():
    scores = np.array([0.9, 0.2, 0.6, 0.4, 0.8, 0.1])
    risk = np.array([2, 2, 2, 1, 1, 0])
    t = 0.5
    assert mdir(scores, risk, t) == pytest.approx(1 / 3)  # high: 0.9, 0.6 flagged, 0.2 missed
    rec = recall_by_risk(scores, risk, t)
    assert rec == {"low": 0.0, "moderate": 0.5, "high": pytest.approx(2 / 3)}
    # (1*0 + 2*0.5 + 3*2/3) / 6 = 3/6
    assert risk_weighted_recall(scores, risk, t) == pytest.approx(0.5)
    assert mdir(scores[:1], risk[:1], 0.95) == 1.0  # scores below t are missed; a tie at t is flagged
    assert mdir([0.5], [2], 0.5) == 0.0


def test_empty_risk_class_is_nan_and_skipped_in_weighted_recall():
    scores, risk = np.array([0.9, 0.1]), np.array([2, 2])
    assert math.isnan(mdir(scores, risk, 0.5, risk=0))
    rec = recall_by_risk(scores, risk, 0.5)
    assert math.isnan(rec["low"]) and rec["high"] == 0.5
    assert risk_weighted_recall(scores, risk, 0.5) == pytest.approx(0.5)


def test_cch_counts_hidden_high_risk_and_half_of_new_alarms():
    t = 0.5
    risk = np.array([2, 2, 2, 1])
    without = np.array([0.7, 0.6, 0.3, 0.9])
    with_ = np.array([0.4, 0.8, 0.6, 0.1])  # pair 0 hidden; pair 2 newly found (not harm); pair 3 is moderate
    neg_without = np.array([0.1, 0.6, 0.2, 0.3])
    neg_with = np.array([0.7, 0.6, 0.9, 0.1])  # negatives 0 and 2 newly flagged; 1 was already flagged
    assert cch(with_, without, risk, neg_with, neg_without, t) == pytest.approx(1 + 0.5 * 2)
    assert cch(with_, without, risk, neg_with, neg_without, t, lam=0.0) == 1.0
    assert cch(without, without, risk, neg_without, neg_without, t) == 0.0  # no effect, no harm
    with pytest.raises(ValueError):
        cch(with_[:2], without, risk, neg_with, neg_without, t)


def test_drr():
    # FedAvg drops from 0.95 to 0.90; the defense recovers to 0.94: 0.04 / 0.05.
    assert drr(0.94, 0.90, 0.95, 0.005) == pytest.approx(0.8)
    assert drr(0.88, 0.90, 0.95, 0.005) == pytest.approx(-0.4)  # worse than FedAvg
    assert math.isnan(drr(0.94, 0.948, 0.95, 0.005))  # attack too weak
    assert math.isnan(drr(0.94, float("nan"), 0.95, 0.005))
