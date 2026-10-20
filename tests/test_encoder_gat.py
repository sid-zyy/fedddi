"""The dense GAT path (used by the pilot for speed) must match PyG's GATConv, duplicates and self-loops included."""

import torch

from src.models.encoder import GraphSAGEEncoder


def test_dense_gat_matches_sparse_gatconv():
    torch.manual_seed(0)
    n = 30
    ei = torch.randint(0, n, (2, 120))  # random edges include duplicates and some self-loops
    ei = torch.cat([ei, ei.flip(0)], 1)
    enc = GraphSAGEEncoder(n, 16, 2, 0.0, False, torch.randn(n, 8), conv="gat").eval()
    for c in enc.convs:  # non-zero attention vectors and bias, so the check is not trivial
        torch.nn.init.normal_(c.att_src)
        torch.nn.init.normal_(c.att_dst)
        torch.nn.init.normal_(c.bias)
    sparse = enc(ei)
    enc.use_dense_adj = True
    assert torch.allclose(sparse, enc(ei), atol=1e-5)
