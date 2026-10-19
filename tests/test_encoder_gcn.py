"""The dense GCN path (used by the pilot for speed) must match PyG's GCNConv."""

import torch

from src.models.encoder import GraphSAGEEncoder


def test_dense_gcn_matches_sparse_gcnconv():
    torch.manual_seed(0)
    n = 30
    ei = torch.randint(0, n, (2, 120))
    ei = torch.cat([ei, ei.flip(0)], 1)
    enc = GraphSAGEEncoder(n, 16, 2, 0.0, False, torch.randn(n, 8), conv="gcn").eval()
    sparse = enc(ei)
    enc.use_dense_adj = True
    assert torch.allclose(sparse, enc(ei), atol=1e-5)
