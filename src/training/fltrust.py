"""FLTrust baseline (Cao et al., NDSS 2021). The server trains on its clean root edges to obtain a
reference update, scores each client by the ReLU-clipped cosine similarity to it, rescales each client
update to the reference norm, and takes the trust-weighted mean. Sample counts are not used.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

NDArrays = List[np.ndarray]


@dataclass
class FLTrustReport:
    server_round: int
    client_keys: List[str]
    cosine: List[float]
    trust: List[float]
    client_norms: List[float]
    server_norm: float

    def as_row_fields(self) -> Dict[str, Any]:
        return {
            "fltrust_client_keys": json.dumps(self.client_keys),
            "fltrust_cosine": json.dumps([round(c, 4) for c in self.cosine]),
            "fltrust_trust": json.dumps([round(t, 4) for t in self.trust]),
            "fltrust_client_norms": json.dumps([round(n, 4) for n in self.client_norms]),
            "fltrust_server_norm": round(self.server_norm, 5),
        }


def _flat_delta(old: NDArrays, new: NDArrays) -> np.ndarray:
    return np.concatenate([(n.astype(np.float64) - o.astype(np.float64)).ravel() for o, n in zip(old, new)])


def fltrust_aggregate(
    old_arrays: NDArrays,
    client_arrays: List[NDArrays],
    server_arrays: NDArrays,
    client_keys: List[str],
    server_round: int = 0,
) -> "tuple[NDArrays, FLTrustReport]":
    """θ + Σ_i TS_i·(‖g0‖/‖g_i‖)·g_i / Σ_i TS_i, with TS_i = ReLU(cos(g_i, g0)).

    If every trust score is zero, or the server update is zero, the global model is left unchanged.
    """
    g0 = _flat_delta(old_arrays, server_arrays)
    n0 = float(np.linalg.norm(g0))

    cosines: List[float] = []
    norms: List[float] = []
    for arrs in client_arrays:
        g = _flat_delta(old_arrays, arrs)
        ni = float(np.linalg.norm(g))
        norms.append(ni)
        cosines.append(float(g @ g0) / (ni * n0) if ni > 0.0 and n0 > 0.0 else 0.0)
    trust = [max(c, 0.0) for c in cosines]
    total = sum(trust)

    new_arrays = [o.copy() for o in old_arrays]
    if total > 0.0:
        for arrs, ts, ni in zip(client_arrays, trust, norms):
            if ts == 0.0:
                continue
            coef = ts * (n0 / ni) / total
            for k, (o, a) in enumerate(zip(old_arrays, arrs)):
                new_arrays[k] = new_arrays[k] + coef * (a.astype(np.float64) - o.astype(np.float64))
    new_arrays = [a.astype(o.dtype, copy=False) for a, o in zip(new_arrays, old_arrays)]

    report = FLTrustReport(server_round=server_round, client_keys=list(client_keys), cosine=cosines,
                           trust=trust, client_norms=norms, server_norm=n0)
    logger.info(
        "[FLTrust] Round %d — server norm=%.4f | cos=%s | trust=%s",
        server_round, n0, [f"{c:.3f}" for c in cosines], [f"{t:.3f}" for t in trust],
    )
    return new_arrays, report
