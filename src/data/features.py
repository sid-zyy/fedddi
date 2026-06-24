"""
src/data/features.py
─────────────────────
Node feature computation for FedDDI-GNN.

Provides two modes:
1. Anchor-based distance encoding (landmark distances) — positional encodings
   that are topology-derived and globally consistent across all clients.

2. Fixed structural node features for Strategy 1 (Feature Bootstrapping):
   compute_fixed_global_features() returns a frozen [N, D] tensor that every
   client and the server can compute identically from the PUBLIC graph topology
   and chemical-free degree statistics.  Because these features are fixed (not
   learned) and topology-derived (not private), they create a shared Euclidean
   coordinate frame for all clients without leaking any learned embedding.

   Feature composition (all topology-only, no chemical data required):
     • Degree-normalized:   log(1 + deg(v)) / log(1 + max_deg)           [1]
     • Degree percentile:   rank(deg(v)) / N                              [1]
     • Anchor distances:    hop dist to `num_anchors` random anchors      [A]
     • Spectral approx:     top-K eigenvector components of normalised L  [K]
                            (computed via sparse power iteration / LOBPCG)
   Total dim D = 2 + A + K (default: 2 + 64 + 32 = 98 → projected to
   hidden_dim=500 via a shared feature_proj Linear that *IS* aggregated).

   Privacy note: all features are derivable from the TRAINING edge_index
   which is already public/shared — no client graph topology is newly exposed.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Sentinel for "unreachable" — large enough to dominate min/sum/diff terms
# without producing inf/nan downstream.
UNREACHABLE_DIST = 32.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal BFS helpers (shared by distance encoding and fixed features)
# ─────────────────────────────────────────────────────────────────────────────


def _multi_source_bfs(adj: List[List[int]], sources: List[int], num_nodes: int) -> Tensor:
    """Run BFS from every node in `sources` simultaneously.

    Returns a FloatTensor [num_nodes] of hop-distance from the *nearest*
    starting point — but since we need a separate distance per anchor (not
    just the nearest), this helper is called once per anchor rather than
    once for all anchors together. See compute_distance_encoding.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Legacy: topological features (degree + PageRank) — server diagnostic only
# ─────────────────────────────────────────────────────────────────────────────


def compute_topo_features(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Compute lightweight topological node features: degree + PageRank.

    LEGACY — kept for backward compatibility with the server diagnostic path.
    For Strategy 1 (Feature Bootstrapping), use compute_fixed_global_features()
    which produces a richer, anchored, spectrally-aware feature set and is the
    correct foundation for a shared coordinate frame across all FL clients.

    Returns FloatTensor [num_nodes, 2] — z-scored degree and PageRank.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Fixed Global Features — the shared coordinate frame
# ─────────────────────────────────────────────────────────────────────────────


def compute_fixed_global_features(
    edge_index: Tensor,
    num_nodes: int,
    num_anchors: int = 64,
    num_spectral: int = 32,
    seed: int = 0,
) -> Tensor:
    """Compute fixed, public, globally-consistent node features.

    These features form the *shared coordinate frame* for Strategy 1
    (Feature Bootstrapping). Every client and the server compute them
    identically from the same public training edge_index, so the resulting
    [N, D] matrix is bit-for-bit identical across all parties. There is no
    learned component — the tensor is registered as a buffer in the encoder,
    not as a parameter — so FedAvg never needs to aggregate it. The shared
    coordinate frame arises purely from the fixed feature space, not from
    any communicated gradient.

    Feature layout:
      [0]      log-degree feature (normalized, 1-dim)
      [1]      degree percentile rank (1-dim)
      [2:2+A]  anchor hop-distances, clipped and z-scored (A = num_anchors)
      [2+A:]   approximate spectral eigenvectors via LOBPCG (K = num_spectral)

    Total dimension: D = 2 + num_anchors + num_spectral
    Default D = 2 + 64 + 32 = 98 → projected to hidden_dim via feature_proj.

    Parameters
    ----------
    edge_index   : LongTensor [2, E] on CPU — public training graph.
    num_nodes    : int
    num_anchors  : int — number of BFS-distance anchor nodes (default 64).
    num_spectral : int — number of approximate Laplacian eigenvectors (default 32).
    seed         : int — reproducible anchor selection.

    Returns
    -------
    FloatTensor [num_nodes, 2 + num_anchors + num_spectral]
    """
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

    # ── 1. Degree features ────────────────────────────────────────────────────
    src_np = edge_index_cpu[0].numpy()
    degree = np.bincount(src_np, minlength=num_nodes).astype(np.float32)
    max_deg = float(degree.max()) if degree.max() > 0 else 1.0

    log_deg = np.log1p(degree) / np.log1p(max_deg)           # [N] in [0,1]
    deg_rank = np.argsort(np.argsort(degree)).astype(np.float32) / max(num_nodes - 1, 1)  # [N] in [0,1]

    # ── 2. Anchor hop-distances ───────────────────────────────────────────────
    effective_anchors = min(num_anchors, num_nodes)
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randperm(num_nodes, generator=generator)[:effective_anchors].tolist()

    adj_list = _build_adjacency(edge_index_cpu, num_nodes)
    dist_mat = np.empty((num_nodes, effective_anchors), dtype=np.float32)
    for k, anchor in enumerate(anchors):
        d = _multi_source_bfs(adj_list, [anchor], num_nodes).numpy()
        dist_mat[:, k] = d

    # Clip unreachable to max finite distance, then z-score
    finite_mask = dist_mat < UNREACHABLE_DIST
    max_finite = float(dist_mat[finite_mask].max()) if finite_mask.any() else 1.0
    dist_mat = np.clip(dist_mat, 0.0, max_finite)
    dist_mean = dist_mat.mean(axis=0, keepdims=True)
    dist_std  = dist_mat.std(axis=0, keepdims=True) + 1e-8
    dist_mat  = (dist_mat - dist_mean) / dist_std

    # ── 3. Approximate spectral features (LOBPCG on normalised Laplacian) ────
    # Symmetric normalised Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    # The bottom-K non-trivial eigenvectors encode global graph geometry and are
    # completely determined by the graph structure — they are the same on every
    # machine given the same edge_index.
    effective_spectral = min(num_spectral, num_nodes - 2)  # must be < N-1
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

    # L_sym = I - A_norm
    L_sym = sp.eye(num_nodes, format="csr") - A_norm

    try:
        # LOBPCG finds the SMALLEST eigenvalues; skip λ=0 (constant vector)
        rng_np = np.random.default_rng(seed)
        X0 = rng_np.standard_normal((num_nodes, effective_spectral)).astype(np.float64)
        # Orthogonalise against the all-ones vector (trivial eigenvector)
        ones = np.ones((num_nodes, 1), dtype=np.float64) / np.sqrt(num_nodes)
        X0 -= ones @ (ones.T @ X0)
        X0, _ = np.linalg.qr(X0)

        eigvecs = spla.lobpcg(
            L_sym,
            X0,
            largest=False,
            maxiter=200,
            tol=1e-6,
        )[1]  # [N, K]
        eigvecs = eigvecs.astype(np.float32)
        # Sign-normalise: flip so first non-zero component is positive (reproducible)
        for k in range(eigvecs.shape[1]):
            first_nonzero = eigvecs[:, k][np.abs(eigvecs[:, k]) > 1e-8]
            if len(first_nonzero) > 0 and first_nonzero[0] < 0:
                eigvecs[:, k] *= -1.0
    except Exception as e:
        logger.warning("LOBPCG failed (%s) — using zeros for spectral features.", e)
        eigvecs = np.zeros((num_nodes, effective_spectral), dtype=np.float32)

    # Pad to exactly num_spectral columns if LOBPCG returned fewer
    if eigvecs.shape[1] < num_spectral:
        pad = np.zeros((num_nodes, num_spectral - eigvecs.shape[1]), dtype=np.float32)
        eigvecs = np.concatenate([eigvecs, pad], axis=1)

    # ── 4. Concatenate and return ─────────────────────────────────────────────
    feat = np.concatenate(
        [
            log_deg[:, None],   # [N, 1]
            deg_rank[:, None],  # [N, 1]
            dist_mat,           # [N, num_anchors]
            eigvecs,            # [N, num_spectral]
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


# ─────────────────────────────────────────────────────────────────────────────
# Distance encoding (unchanged, used by optional dist_enc path)
# ─────────────────────────────────────────────────────────────────────────────


def compute_distance_encoding(
    edge_index: Tensor,
    num_nodes: int,
    num_anchors: int = 200,
    seed: int = 0,
) -> Tensor:
    """Compute anchor/landmark distance encodings for every node.

    Parameters
    ----------
    edge_index  : LongTensor [2, E] — undirected graph edge index (CPU).
    num_nodes   : int
    num_anchors : int — number of randomly sampled anchor nodes.
    seed        : int — RNG seed.

    Returns
    -------
    dist_enc : FloatTensor [num_nodes, num_anchors]
    """
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
