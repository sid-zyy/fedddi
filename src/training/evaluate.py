"""OGB-compatible evaluation for ogbl-ddi"""

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
    """Run evaluation using the OGB evaluator"""
    encoder.eval()
    predictor.eval()

    h = encoder(edge_index.to(device))

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


@torch.no_grad()
def score_edge_set(
    encoder: GraphSAGEEncoder,
    predictor: NeuralLinkPredictor,
    edge_index: Tensor,
    edges: Tensor,
    batch_size: int = 65536,
    device: torch.device | str = "cpu",
    node_dist_enc: Tensor | None = None,
) -> dict[str, float]:
    """Score an arbitrary, un-ranked set of edges and summarize how confidently the CURRENT model
    believes each one is a true interaction
    """
    if edges.numel() == 0:
        return {"mean_prob": 0.0, "n": 0}

    encoder.eval()
    predictor.eval()

    h = encoder(edge_index.to(device))
    logits = score_edges(h, predictor, edges, batch_size, device, node_dist_enc)
    probs = torch.sigmoid(logits)
    return {"mean_prob": float(probs.mean().item()), "n": int(edges.size(0))}


def score_edges(
    h: Tensor,
    predictor: NeuralLinkPredictor,
    edges: Tensor,
    batch_size: int,
    device: torch.device | str,
    node_dist_enc: Tensor | None = None,
) -> Tensor:
    """Score a set of edges in mini-batches"""
    scores: list[Tensor] = []
    edges = edges.to(device)

    for start in range(0, edges.size(0), batch_size):
        end = min(start + batch_size, edges.size(0))
        batch = edges[start:end]
        src, dst = batch[:, 0], batch[:, 1]

        d_u, d_v = None, None
        if node_dist_enc is not None:
            d_u, d_v = node_dist_enc[src], node_dist_enc[dst]

        logits = predictor(h[src], h[dst], d_u, d_v)
        scores.append(logits.squeeze(-1).cpu())

    return torch.cat(scores, dim=0)
