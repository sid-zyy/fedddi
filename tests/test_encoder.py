"""Unit tests for GraphSAGEEncoder"""

from __future__ import annotations

import torch
import pytest

from src.models.encoder import GraphSAGEEncoder
from tests.conftest import HIDDEN_DIM, NUM_NODES


@pytest.fixture
def encoder():
    return GraphSAGEEncoder(
        num_nodes=NUM_NODES,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.3,
        normalize=True,
    )


class TestGraphSAGEEncoder:
    def test_output_shape(self, encoder, edge_index):
        h = encoder(edge_index)
        assert h.shape == (NUM_NODES, HIDDEN_DIM)

    def test_output_dtype(self, encoder, edge_index):
        h = encoder(edge_index)
        assert h.dtype == torch.float32

    def test_l2_norm_when_normalize_true(self, encoder, edge_index):
        h = encoder(edge_index)
        norms = h.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(NUM_NODES), atol=1e-5, rtol=0)

    def test_no_l2_norm_when_normalize_false(self, edge_index):
        enc = GraphSAGEEncoder(
            num_nodes=NUM_NODES,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            dropout=0.0,
            normalize=False,
        )
        enc.eval()
        h = enc(edge_index)
        norms = h.norm(dim=-1)

        assert not torch.allclose(norms, torch.ones(NUM_NODES), atol=1e-3)

    def test_embedding_is_parameter(self, encoder):
        param_names = [n for n, _ in encoder.named_parameters()]
        assert "embedding.weight" in param_names

    def test_embedding_shape(self, encoder):
        assert encoder.embedding.weight.shape == (NUM_NODES, HIDDEN_DIM)

    def test_gradient_flows_to_embedding(self, encoder, edge_index):
        encoder.train()
        h = encoder(edge_index)
        loss = h.sum()
        loss.backward()
        assert encoder.embedding.weight.grad is not None
        assert encoder.embedding.weight.grad.abs().sum() > 0

    def test_gradient_flows_to_conv_weights(self, encoder, edge_index):
        encoder.train()
        h = encoder(edge_index)
        loss = h.sum()
        loss.backward()
        for i, conv in enumerate(encoder.convs):
            for name, param in conv.named_parameters():
                assert param.grad is not None, f"conv[{i}].{name} has no gradient"

    def test_train_eval_differ_with_dropout(self, encoder, edge_index):
        """Dropout should cause stochastic outputs in train mode."""
        torch.manual_seed(0)
        encoder.train()
        h_train_1 = encoder(edge_index).detach().clone()
        h_train_2 = encoder(edge_index).detach().clone()

        encoder.eval()
        h_eval_1 = encoder(edge_index).detach().clone()
        h_eval_2 = encoder(edge_index).detach().clone()


        torch.testing.assert_close(h_eval_1, h_eval_2)




        assert not torch.allclose(h_train_1, h_train_2, atol=1e-7)

    def test_reproducibility(self, edge_index):
        """Same seed → same output (eval mode, no dropout)."""
        torch.manual_seed(42)
        enc1 = GraphSAGEEncoder(NUM_NODES, HIDDEN_DIM, 2, 0.0, True)
        enc1.eval()

        torch.manual_seed(42)
        enc2 = GraphSAGEEncoder(NUM_NODES, HIDDEN_DIM, 2, 0.0, True)
        enc2.eval()

        h1 = enc1(edge_index)
        h2 = enc2(edge_index)
        torch.testing.assert_close(h1, h2)

    def test_different_seeds_produce_different_outputs(self, edge_index):
        torch.manual_seed(0)
        enc1 = GraphSAGEEncoder(NUM_NODES, HIDDEN_DIM, 2, 0.0, True)
        enc1.eval()

        torch.manual_seed(99)
        enc2 = GraphSAGEEncoder(NUM_NODES, HIDDEN_DIM, 2, 0.0, True)
        enc2.eval()

        h1 = enc1(edge_index)
        h2 = enc2(edge_index)
        assert not torch.allclose(h1, h2)

    def test_num_layers_respected(self, edge_index):
        for nl in [1, 2, 3]:
            enc = GraphSAGEEncoder(NUM_NODES, HIDDEN_DIM, nl, 0.0, False)
            assert len(enc.convs) == nl
            h = enc(edge_index)
            assert h.shape == (NUM_NODES, HIDDEN_DIM)

    def test_parameter_count_scales_with_layers(self):
        enc1 = GraphSAGEEncoder(NUM_NODES, HIDDEN_DIM, 1, 0.0, False)
        enc2 = GraphSAGEEncoder(NUM_NODES, HIDDEN_DIM, 3, 0.0, False)
        p1 = sum(p.numel() for p in enc1.parameters())
        p2 = sum(p.numel() for p in enc2.parameters())
        assert p2 > p1
