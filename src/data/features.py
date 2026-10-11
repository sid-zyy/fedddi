"""Node feature computation for FedDDI-GNN"""

from __future__ import annotations

import logging
from collections import deque
from typing import List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


UNREACHABLE_DIST = 32.0


def _multi_source_bfs(adj: List[List[int]], sources: List[int], num_nodes: int) -> Tensor:
    """Run BFS from every node in `sources` simultaneously"""
    dist = torch.full((num_nodes,), UNREACHABLE_DIST, dtype=torch.float32)
    visited = torch.zeros(num_nodes, dtype=torch.bool)

    q: deque[int] = deque()
    for s in sources:
        dist[s] = 0.0
        visited[s] = True
        q.append(s)

    while q:
        u = q.popleft()
        d = dist[u].item()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                dist[v] = d + 1.0
                q.append(v)

    return dist


def _build_adjacency(edge_index: Tensor, num_nodes: int) -> List[List[int]]:
    """Build a Python adjacency list for fast BFS."""
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        adj[u].append(v)
    return adj


def compute_topo_features(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Compute lightweight topological node features: degree + PageRank"""
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp  # type: ignore

    logger.info("Computing topological features (degree + PageRank) ...")

    edge_index_cpu = edge_index.cpu()
    adj_scipy = sp.coo_matrix(
        (
            np.ones(edge_index_cpu.size(1)),
            (edge_index_cpu[0].numpy(), edge_index_cpu[1].numpy()),
        ),
        shape=(num_nodes, num_nodes),
    ).tocsr()

    degrees = np.array(adj_scipy.sum(axis=1)).squeeze()

    graph = nx.from_scipy_sparse_array(adj_scipy)
    pagerank = nx.pagerank(graph)
    pageranks = np.array([pagerank[i] for i in range(num_nodes)])

    topo_feat = np.stack([degrees, pageranks], axis=1)
    topo_feat = (topo_feat - topo_feat.mean(axis=0)) / (topo_feat.std(axis=0) + 1e-8)

    return torch.tensor(topo_feat, dtype=torch.float32)


def compute_fixed_global_features(
    edge_index: Tensor,
    num_nodes: int,
    num_anchors: int = 64,
    num_spectral: int = 32,
    seed: int = 0,
) -> Tensor:
    """Compute fixed, public, globally-consistent node features"""
    import numpy as np
    import scipy.sparse as sp  # type: ignore
    import scipy.sparse.linalg as spla  # type: ignore

    logger.info(
        "Computing fixed global features — %d anchors + %d spectral dims over %d nodes ...",
        num_anchors,
        num_spectral,
        num_nodes,
    )
    edge_index_cpu = edge_index.cpu()


    src_np = edge_index_cpu[0].numpy()
    degree = np.bincount(src_np, minlength=num_nodes).astype(np.float32)
    max_deg = float(degree.max()) if degree.max() > 0 else 1.0

    log_deg = np.log1p(degree) / np.log1p(max_deg)
    deg_rank = np.argsort(np.argsort(degree)).astype(np.float32) / max(num_nodes - 1, 1)


    effective_anchors = min(num_anchors, num_nodes)
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randperm(num_nodes, generator=generator)[:effective_anchors].tolist()

    adj_list = _build_adjacency(edge_index_cpu, num_nodes)
    dist_mat = np.empty((num_nodes, effective_anchors), dtype=np.float32)
    for k, anchor in enumerate(anchors):
        d = _multi_source_bfs(adj_list, [anchor], num_nodes).numpy()
        dist_mat[:, k] = d


    finite_mask = dist_mat < UNREACHABLE_DIST
    max_finite = float(dist_mat[finite_mask].max()) if finite_mask.any() else 1.0
    dist_mat = np.clip(dist_mat, 0.0, max_finite)
    dist_mean = dist_mat.mean(axis=0, keepdims=True)
    dist_std  = dist_mat.std(axis=0, keepdims=True) + 1e-8
    dist_mat  = (dist_mat - dist_mean) / dist_std






    effective_spectral = min(num_spectral, num_nodes - 2)
    effective_spectral = max(1, effective_spectral)

    row = np.concatenate([src_np, edge_index_cpu[1].numpy()])
    col = np.concatenate([edge_index_cpu[1].numpy(), src_np])
    data = np.ones(len(row), dtype=np.float64)
    A_sp = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A_sp.sum_duplicates()

    deg_arr = np.array(A_sp.sum(axis=1)).squeeze().astype(np.float64)
    deg_arr = np.maximum(deg_arr, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_arr)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_sp @ D_inv_sqrt


    L_sym = sp.eye(num_nodes, format="csr") - A_norm

    try:

        rng_np = np.random.default_rng(seed)
        X0 = rng_np.standard_normal((num_nodes, effective_spectral)).astype(np.float64)

        ones = np.ones((num_nodes, 1), dtype=np.float64) / np.sqrt(num_nodes)
        X0 -= ones @ (ones.T @ X0)
        X0, _ = np.linalg.qr(X0)

        eigvecs = spla.lobpcg(
            L_sym,
            X0,
            largest=False,
            maxiter=200,
            tol=1e-6,
        )[1]
        eigvecs = eigvecs.astype(np.float32)

        for k in range(eigvecs.shape[1]):
            first_nonzero = eigvecs[:, k][np.abs(eigvecs[:, k]) > 1e-8]
            if len(first_nonzero) > 0 and first_nonzero[0] < 0:
                eigvecs[:, k] *= -1.0
    except Exception as e:
        logger.warning("LOBPCG failed (%s) — using zeros for spectral features.", e)
        eigvecs = np.zeros((num_nodes, effective_spectral), dtype=np.float32)


    if eigvecs.shape[1] < num_spectral:
        pad = np.zeros((num_nodes, num_spectral - eigvecs.shape[1]), dtype=np.float32)
        eigvecs = np.concatenate([eigvecs, pad], axis=1)


    feat = np.concatenate(
        [
            log_deg[:, None],
            deg_rank[:, None],
            dist_mat,
            eigvecs,
        ],
        axis=1,
    ).astype(np.float32)

    logger.info(
        "Fixed global features computed — shape %s (degree:2 + anchors:%d + spectral:%d).",
        feat.shape,
        effective_anchors,
        min(num_spectral, num_nodes - 2),
    )
    return torch.tensor(feat, dtype=torch.float32)


def compute_distance_encoding(
    edge_index: Tensor,
    num_nodes: int,
    num_anchors: int = 200,
    seed: int = 0,
) -> Tensor:
    """Compute anchor/landmark distance encodings for every node"""
    if num_anchors <= 0:
        raise ValueError(f"num_anchors must be > 0, got {num_anchors}")
    if num_anchors > num_nodes:
        logger.warning(
            "num_anchors (%d) > num_nodes (%d); clamping to num_nodes.",
            num_anchors,
            num_nodes,
        )
        num_anchors = num_nodes

    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randperm(num_nodes, generator=generator)[:num_anchors].tolist()

    logger.info(
        "Computing anchor distance encoding — %d anchors over %d nodes ...",
        num_anchors,
        num_nodes,
    )

    adj = _build_adjacency(edge_index, num_nodes)

    dist_enc = torch.empty((num_nodes, num_anchors), dtype=torch.float32)
    for k, anchor in enumerate(anchors):
        dist_enc[:, k] = _multi_source_bfs(adj, [anchor], num_nodes)

    logger.info(
        "Distance encoding computed — shape %s, %% unreachable entries: %.2f%%",
        tuple(dist_enc.shape),
        100.0 * (dist_enc >= UNREACHABLE_DIST).float().mean().item(),
    )

    return dist_enc
