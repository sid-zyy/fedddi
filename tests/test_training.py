"""Integration tests for training loop and evaluation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling  # type: ignore

from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor
from src.training.evaluate import score_edges

HIDDEN_DIM = 32
NUM_NODES = 50


@pytest.fixture
def encoder():
    return GraphSAGEEncoder(
        num_nodes=NUM_NODES, hidden_dim=HIDDEN_DIM, num_layers=2, dropout=0.0
    )


@pytest.fixture
def predictor():
    return NeuralLinkPredictor(
        emb_dim=HIDDEN_DIM, hidden_dim=64, num_layers=2, dropout=0.0
    )


class TestForwardPass:
    def test_encoder_predictor_chain(
        self, encoder, predictor, edge_index, synthetic_split
    ):
        encoder.eval()
        predictor.eval()
        h = encoder(edge_index)
        pos = synthetic_split["train"]["edge"][:8]
        src, dst = pos[:, 0], pos[:, 1]
        logits = predictor(h[src], h[dst])
        assert logits.shape == (8, 1)

    def test_loss_is_finite(self, encoder, predictor, edge_index, synthetic_split):
        encoder.train()
        predictor.train()
        h = encoder(edge_index)
        pos = synthetic_split["train"]["edge"][:16]
        src, dst = pos[:, 0], pos[:, 1]
        logits_p = predictor(h[src], h[dst])
        neg_ei = negative_sampling(edge_index, num_nodes=NUM_NODES, num_neg_samples=16)
        logits_n = predictor(h[neg_ei[0]], h[neg_ei[1]])
        loss = F.binary_cross_entropy_with_logits(
            logits_p, torch.ones_like(logits_p)
        ) + F.binary_cross_entropy_with_logits(logits_n, torch.zeros_like(logits_n))
        assert torch.isfinite(loss)

    def test_backward_updates_params(
        self, encoder, predictor, edge_index, synthetic_split
    ):
        encoder.train()
        predictor.train()
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
        )
        h = encoder(edge_index)
        pos = synthetic_split["train"]["edge"][:16]
        logits_p = predictor(h[pos[:, 0]], h[pos[:, 1]])
        neg_ei = negative_sampling(edge_index, num_nodes=NUM_NODES, num_neg_samples=16)
        logits_n = predictor(h[neg_ei[0]], h[neg_ei[1]])
        loss = F.binary_cross_entropy_with_logits(
            logits_p, torch.ones_like(logits_p)
        ) + F.binary_cross_entropy_with_logits(logits_n, torch.zeros_like(logits_n))
        params_before = [p.clone() for p in encoder.parameters()]
        opt.zero_grad()
        loss.backward()
        opt.step()
        for p_old, p_new in zip(params_before, encoder.parameters()):
            if p_new.grad is not None and p_new.grad.abs().sum() > 0:
                assert not torch.equal(p_old, p_new)
                break


class TestScoreEdges:
    def test_score_shape(self, encoder, predictor, edge_index, synthetic_split):
        encoder.eval()
        predictor.eval()
        h = encoder(edge_index)
        edges = synthetic_split["valid"]["edge"]
        scores = score_edges(h, predictor, edges, 32, "cpu")
        assert scores.shape == (edges.size(0),)
        assert scores.dtype == torch.float32

    def test_score_range(self, encoder, predictor, edge_index, synthetic_split):
        encoder.eval()
        predictor.eval()
        h = encoder(edge_index)
        edges = synthetic_split["valid"]["edge"]
        scores = score_edges(h, predictor, edges, 32, "cpu")
        assert torch.all(torch.isfinite(scores))
