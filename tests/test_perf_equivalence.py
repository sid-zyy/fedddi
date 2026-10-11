"""Run BEFORE enabling perf.dense_adj / perf.fast_negatives (needs torch + torch_geometric)"""
import torch
from src.models.encoder import GraphSAGEEncoder


def _graph(N=300, E=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, N, (2, E), generator=g)
    ei[:, :30] = ei[:, [0]]
    ei[1][ei[1] == 7] = 8
    return ei


def test_dense_matches_sparse_forward_and_grad():
    N = 300
    ei = _graph(N)
    feats = torch.randn(N, 16)
    enc = GraphSAGEEncoder(N, hidden_dim=32, num_layers=2, dropout=0.0, node_features=feats)
    enc.eval()
    enc.use_dense_adj = False
    h_sparse = enc(ei)
    h_sparse.square().sum().backward()
    g_sparse = {n: p.grad.clone() for n, p in enc.named_parameters()}
    enc.zero_grad()
    enc.use_dense_adj = True
    h_dense = enc(ei)
    h_dense.square().sum().backward()
    assert torch.allclose(h_sparse, h_dense, atol=1e-4, rtol=1e-4)
    for n, p in enc.named_parameters():
        assert torch.allclose(g_sparse[n], p.grad, atol=1e-3, rtol=1e-3), n


def test_parameter_names_and_order_unchanged():
    enc = GraphSAGEEncoder(50, hidden_dim=8, num_layers=2, node_features=torch.randn(50, 4))
    names_before = [n for n, _ in enc.named_parameters()]
    enc.use_dense_adj = True
    enc(_graph(50, 300))
    assert [n for n, _ in enc.named_parameters()] == names_before


def test_fast_negatives_exclude_positives_and_are_in_range():
    from types import SimpleNamespace
    from src.training.trainer import Trainer
    N = 200
    ei = _graph(N, 3000)
    t = Trainer.__new__(Trainer)
    t.encoder = SimpleNamespace(num_nodes=N)
    t.edge_index, t.device, t._fast_neg, t._edge_keys = ei, torch.device("cpu"), True, None
    neg = t._sample_negatives(5000)
    assert neg.shape == (2, 5000) and neg.min() >= 0 and neg.max() < N
    pos = set((ei[0] * N + ei[1]).tolist())
    assert not any(int(u) * N + int(v) in pos for u, v in neg.t())
