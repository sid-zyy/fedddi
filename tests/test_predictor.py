"""Tests for NeuralLinkPredictor (Hadamard MLP decoder)."""

from __future__ import annotations

import pytest
import torch
from src.models.predictor import NeuralLinkPredictor

HIDDEN_DIM = 32
BATCH = 16


@pytest.fixture
def predictor():


    torch.manual_seed(0)
    return NeuralLinkPredictor(
        emb_dim=HIDDEN_DIM, hidden_dim=64, num_layers=3, dropout=0.0
    )


@pytest.fixture
def rand_embeddings():



    torch.manual_seed(1)
    h_u = torch.rand(BATCH, HIDDEN_DIM)
    h_v = torch.rand(BATCH, HIDDEN_DIM)
    return h_u, h_v


class TestNeuralLinkPredictor:
    def test_output_shape(self, predictor, rand_embeddings):
        h_u, h_v = rand_embeddings
        predictor.eval()
        out = predictor(h_u, h_v)
        assert out.shape == (BATCH, 1)

    def test_output_dtype(self, predictor, rand_embeddings):
        h_u, h_v = rand_embeddings
        predictor.eval()
        out = predictor(h_u, h_v)
        assert out.dtype == torch.float32

    def test_logits_are_unbounded(self, predictor, rand_embeddings):



        h_u, h_v = rand_embeddings
        predictor.eval()
        out = predictor(h_u, h_v)
        assert not torch.all(out >= 0) or not torch.all(out <= 1)

    def test_sigmoid_gives_valid_probs(self, predictor, rand_embeddings):
        h_u, h_v = rand_embeddings
        predictor.eval()
        probs = torch.sigmoid(predictor(h_u, h_v))
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_gradients_flow(self, predictor, rand_embeddings):
        h_u, h_v = rand_embeddings
        predictor.train()
        logits = predictor(h_u, h_v)
        loss = logits.sum()
        loss.backward()
        for name, param in predictor.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_eval_is_deterministic(self, predictor, rand_embeddings):
        h_u, h_v = rand_embeddings
        predictor.eval()
        out1 = predictor(h_u, h_v)
        out2 = predictor(h_u, h_v)
        assert torch.allclose(out1, out2)

    def test_element_wise_product_symmetry(self, predictor, rand_embeddings):
        """h_u ⊙ h_v == h_v ⊙ h_u so output should be identical."""
        h_u, h_v = rand_embeddings
        predictor.eval()
        out_fwd = predictor(h_u, h_v)
        out_rev = predictor(h_v, h_u)
        assert torch.allclose(out_fwd, out_rev)

    def test_batch_size_one(self, predictor):
        predictor.eval()
        h_u = torch.rand(1, HIDDEN_DIM)
        h_v = torch.rand(1, HIDDEN_DIM)
        out = predictor(h_u, h_v)
        assert out.shape == (1, 1)

    def test_large_batch(self, predictor):
        B = 1024
        predictor.eval()
        out = predictor(torch.rand(B, HIDDEN_DIM), torch.rand(B, HIDDEN_DIM))
        assert out.shape == (B, 1)
