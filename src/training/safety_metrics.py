"""Safety metrics for DDI prediction under poisoning (Paper A, PUBLICATION_PLAN.md "Metric definitions").

Risk codes follow src/data/ddi_risk.py: 0 low, 1 moderate, 2 high. Scores may be logits or
probabilities; every metric here only compares scores against a threshold, so either works as long as
the threshold comes from the same kind of score.

- threshold_at_specificity: the decision threshold t at which a share `spec` of negatives score below t.
- mdir: Missed Dangerous Interaction Rate, the share of true high-risk interactions not flagged (score < t).
- recall_by_risk / risk_weighted_recall: recall per risk class and its weighted mean (weights 1, 2, 3).
- cch: Counterfactual Clinical Harm of one client, in high-risk interactions hidden by including it,
  plus lam times the false alarms it raises.
- drr: Damage Recovery Ratio of a defense against FedAvg, undefined when the attack barely hurts FedAvg.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

HIGH = 2
RISK_NAMES = ("low", "moderate", "high")
DEFAULT_WEIGHTS = (1.0, 2.0, 3.0)


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def threshold_at_specificity(neg_scores, spec: float = 0.95) -> float:
    """Smallest observed negative score t with at least a share `spec` of negatives strictly below it
    (ties aside). A pair is flagged when its score is >= t."""
    neg = np.sort(_np(neg_scores))
    if neg.size == 0:
        raise ValueError("need at least one negative score")
    k = min(int(math.ceil(spec * neg.size)), neg.size - 1)
    return float(neg[k])


def recall_by_risk(pos_scores, pos_risk, t: float) -> Dict[str, float]:
    """Share of positives flagged (score >= t) in each risk class; NaN for an empty class."""
    s, r = _np(pos_scores), _np(pos_risk).astype(np.int64)
    out = {}
    for code, name in enumerate(RISK_NAMES):
        m = r == code
        out[name] = float((s[m] >= t).mean()) if m.any() else float("nan")
    return out


def mdir(pos_scores, pos_risk, t: float, risk: int = HIGH) -> float:
    """Missed Dangerous Interaction Rate: share of positives of class `risk` scoring below t."""
    s, r = _np(pos_scores), _np(pos_risk).astype(np.int64)
    m = r == risk
    return float((s[m] < t).mean()) if m.any() else float("nan")


def risk_weighted_recall(pos_scores, pos_risk, t: float, weights: Sequence[float] = DEFAULT_WEIGHTS) -> float:
    """sum_r w_r * recall_r / sum_r w_r over the classes present."""
    rec = recall_by_risk(pos_scores, pos_risk, t)
    num = den = 0.0
    for code, name in enumerate(RISK_NAMES):
        if not math.isnan(rec[name]):
            num += weights[code] * rec[name]
            den += weights[code]
    return num / den if den else float("nan")


def cch(scores_with, scores_without, risk, neg_with, neg_without, t: float, lam: float = 0.5) -> float:
    """Counterfactual Clinical Harm of one client.

    scores_with / scores_without: scores of the anchor's positives under the aggregate with and without
    the client; risk: their risk codes (only high-risk positives count). neg_with / neg_without: scores
    of the anchor's sampled non-interacting pairs. Both models are judged at the same threshold t.
    Returns (#high-risk positives flagged without the client but not with it)
    + lam * (#negatives flagged with the client but not without it).
    """
    w, wo, r = _np(scores_with), _np(scores_without), _np(risk).astype(np.int64)
    if not (w.size == wo.size == r.size):
        raise ValueError("positive scores and risk must align")
    nw, nwo = _np(neg_with), _np(neg_without)
    if nw.size != nwo.size:
        raise ValueError("negative scores must align")
    h = r == HIGH
    hidden = int(((wo[h] >= t) & (w[h] < t)).sum())
    alarms = int(((nwo < t) & (nw >= t)).sum())
    return float(hidden + lam * alarms)


def drr(def_attack: float, fedavg_attack: float, fedavg_clean: float, min_gap: float) -> float:
    """(defense under attack - FedAvg under attack) / (FedAvg clean - FedAvg under attack), for a
    higher-is-better score. NaN when the attack costs FedAvg less than `min_gap` (0.005 AUROC, or 0.02
    for 1 - MDIR in the plan)."""
    gap = fedavg_clean - fedavg_attack
    if not gap >= min_gap:  # also catches NaN
        return float("nan")
    return float((def_attack - fedavg_attack) / gap)
