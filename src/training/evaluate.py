"""
src/training/evaluate.py
────────────────────────
OGB-compatible evaluation for ogbl-ddi.

Metric: Hits@20
   Given a ranked list of (positive, negative) edge scores, Hits@K is the
   fraction of positive edges whose predicted score is ranked among the top K
   of the combined list.  The OGB evaluator handles this officially.
"""

from __future__ import annotations

import torch
from torch import Tensor
from ogb.linkproppred import Evaluator  # type: ignore
from typing import Any

from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor


@torch.no_grad()
def evaluate(
    encoder: GraphSAGEEncoder,
    predictor: NeuralLinkPredictor,
    edge_index: Tensor,
    pos_edges: Tensor,
    neg_edges: Tensor,
    evaluator: Evaluator,
    batch_size: int = 65536,
    device: torch.device | str = "cpu",
    node_dist_enc: Tensor | None = None,
) -> dict[str, float]:
    """Run evaluation using the OGB evaluator.

    Parameters
    ----------
    encoder    : trained GraphSAGE encoder
    predictor  : trained NeuralLinkPredictor
    edge_index : [2, E] training edges for GNN message passing
    pos_edges  : [N_pos, 2] positive evaluation edges
    neg_edges  : [N_neg, 2] negative evaluation edges
    evaluator  : OGB Evaluator(name="ogbl-ddi")
    batch_size : mini-batch size for score computation
    device     : torch device
    node_dist_enc: optional distance encodings [N, dist_dim]

    Returns
    -------
    dict with key "Hits@20" (float)
    """
    encoder.eval()
    predictor.eval()

    h = encoder(edge_index.to(device))  # [N, D]

    pos_scores = score_edges(h, predictor, pos_edges, batch_size, device, node_dist_enc)
    neg_scores = score_edges(h, predictor, neg_edges, batch_size, device, node_dist_enc)

    result: dict[str, Any] = evaluator.eval(  # type: ignore
        {
            "y_pred_pos": pos_scores,
            "y_pred_neg": neg_scores,
        }
    )
    hits = {k.replace("hits@", "Hits@"): float(v) for k, v in result.items()}
    return hits


def score_edges(
    h: Tensor,
    predictor: NeuralLinkPredictor,
    edges: Tensor,
    batch_size: int,
    device: torch.device | str,
    node_dist_enc: Tensor | None = None,
) -> Tensor:
    """Score a set of edges in mini-batches. Returns 1-D score tensor."""
    scores: list[Tensor] = []
    edges = edges.to(device)

    for start in range(0, edges.size(0), batch_size):
        end = min(start + batch_size, edges.size(0))
        batch = edges[start:end]
        src, dst = batch[:, 0], batch[:, 1]

        d_u, d_v = None, None
        if node_dist_enc is not None:
            d_u, d_v = node_dist_enc[src], node_dist_enc[dst]

        logits = predictor(h[src], h[dst], d_u, d_v)  # [B, 1]
        scores.append(logits.squeeze(-1).cpu())

    return torch.cat(scores, dim=0)
