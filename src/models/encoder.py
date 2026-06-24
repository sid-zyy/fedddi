"""
src/models/encoder.py  (Strategy 1 — Feature Bootstrapping)
─────────────────────────────────────────────────────────────
GraphSAGE encoder for ogbl-ddi.

Architecture
────────────
  [Strategy 1]  fixed_features [N, feat_dim]
       │
  feature_proj Linear(feat_dim → hidden_dim)   ← THIS is now aggregated
       │
  SAGEConv(D, D) + ReLU + Dropout               → h^1  [N, D]
       │
  SAGEConv(D, D)                                 → h^L  [N, D]

  [Legacy / ablation]  nn.Embedding(num_nodes, hidden_dim)  (private, local)

Key change from the original:
  • In the original code, node_features=None caused an nn.Embedding table that
    was NEVER aggregated. The convs and MLP were aggregated, but each client's
    embedding lived in its own private coordinate space — geometrically
    orthogonal to every other client's. Aggregating convs on top of orthogonal
    embeddings is equivalent to averaging random rotations: the result is not
    a useful representation of anything.
  • In Strategy 1, node_features is always provided (the public fixed global
    features). The feature_proj Linear projects these into hidden_dim and IS
    included in the aggregation parameter list. Because every client's
    node_features are bit-for-bit identical (same topology, same seed), the
    feature_proj lives in a shared Euclidean space from round 1, and FedAvg
    correctly averages meaningful gradient directions.
  • The nn.Embedding table is removed in this mode. No private parameters are
    needed — the fixed features already supply every node's initial identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # type: ignore


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder.

    Parameters
    ----------
    num_nodes:      int   — total node count (used for device/size checks).
    hidden_dim:     int   — hidden & output embedding dimension (default 500).
    num_layers:     int   — number of SAGEConv layers (default 2).
    dropout:        float — dropout probability applied after each hidden layer.
    normalize:      bool  — if True, L2-normalise the output embeddings.
    node_features:  Tensor | None
                    If provided (Strategy 1), these FIXED global features are
                    registered as a non-parameter buffer and projected to
                    hidden_dim via feature_proj — which IS aggregated by FedAvg.
                    If None (legacy ablation), a learnable Embedding table is
                    used instead — NOT recommended for federated training.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 500,
        num_layers: int = 2,
        dropout: float = 0.3,
        normalize: bool = False,
        node_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize

        if node_features is not None:
            # ── Strategy 1: fixed public features ───────────────────────────
            # Register as a buffer — moves to device automatically and is
            # saved in state_dict for checkpoint compatibility, but is NOT
            # a Parameter so it does not appear in optimizer updates or
            # federated gradient aggregation.
            self.register_buffer("node_features", node_features.float())
            # feature_proj IS a Parameter: its weights reflect what the
            # federation collectively learned about the feature-to-embedding
            # mapping, and must be aggregated by FedAvg.
            self.feature_proj = nn.Linear(node_features.shape[1], hidden_dim, bias=True)
            self.embedding = None
        else:
            # ── Legacy / ablation: private learnable embedding ───────────────
            # WARNING: in a federated setting this creates orthogonal client
            # coordinate spaces. Only use for ablation studies.
            self.node_features = None  # type: ignore[assignment]
            self.feature_proj = None
            self.embedding = nn.Embedding(num_nodes, hidden_dim)

        # ── SAGEConv stack ─────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.embedding is not None:
            nn.init.xavier_uniform_(self.embedding.weight)
        if self.feature_proj is not None:
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings via message passing over all nodes.

        Parameters
        ----------
        edge_index : LongTensor [2, E]

        Returns
        -------
        h : FloatTensor [num_nodes, hidden_dim]
        """
        if self.node_features is not None:
            assert self.feature_proj is not None
            h = self.feature_proj(self.node_features)   # [N, D]
        else:
            assert self.embedding is not None
            node_ids = torch.arange(self.num_nodes, device=edge_index.device)
            h = self.embedding(node_ids)                # [N, D]

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)                     # [N, D]
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)

        return h  # [N, D]

    @property
    def uses_fixed_features(self) -> bool:
        """True when operating in Strategy-1 (Feature Bootstrapping) mode."""
        return self.node_features is not None
