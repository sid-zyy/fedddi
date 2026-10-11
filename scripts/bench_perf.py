"""Micro-benchmark of the two speed flags at ogbl-ddi scale (synthetic graph; no dataset needed)"""
import time, torch
from torch_geometric.utils import negative_sampling
from src.models.encoder import GraphSAGEEncoder

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, E, D, F = 4267, 2_100_000, 500, 16
ei = torch.randint(0, N, (2, E), device=dev)
enc = GraphSAGEEncoder(N, hidden_dim=D, num_layers=2, node_features=torch.randn(N, F)).to(dev)

def sync():
    if dev.type == "cuda": torch.cuda.synchronize()

def timeit(fn, n=10):
    fn(); sync(); t = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t) / n * 1000

def step():
    enc.zero_grad(); enc(ei).square().mean().backward()

for dense in (False, True):
    enc.use_dense_adj = dense
    print(f"encoder fwd+bwd  dense_adj={dense!s:5}: {timeit(step):8.1f} ms")
keys = torch.unique(ei[0] * N + ei[1])
def fast():
    u = torch.randint(0, N, (85000,), device=dev); v = torch.randint(0, N, (85000,), device=dev)
    k = u * N + v; i = torch.searchsorted(keys, k).clamp_(max=keys.numel() - 1); m = keys[i] != k
    return torch.stack([u[m], v[m]])[:, :65536]
print(f"negatives 65,536  PyG negative_sampling : {timeit(lambda: negative_sampling(ei, num_nodes=N, num_neg_samples=65536)):8.1f} ms")
print(f"negatives 65,536  GPU searchsorted      : {timeit(fast):8.1f} ms")
