"""GraphSAGE encoder for ogbl-ddi (conv="gcn" or "gat" swaps in GCN or GAT layers, the other architectures for Paper A)"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv  # type: ignore


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder"""

    GAT_HEADS = 4

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 500,
        num_layers: int = 2,
        dropout: float = 0.3,
        normalize: bool = False,
        node_features: torch.Tensor | None = None,
        conv: str = "sage",
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        if conv not in ("sage", "gcn", "gat"):
            raise ValueError(f"unknown conv {conv!r}")
        self.conv = conv

        if node_features is not None:





            self.register_buffer("node_features", node_features.float().clone())




            self.register_buffer("_base_node_features", node_features.float().clone())



            self.feature_proj = nn.Linear(node_features.shape[1], hidden_dim, bias=True)
            self.embedding = None
        else:



            self.node_features = None  # type: ignore[assignment]
            self.feature_proj = None
            self.embedding = nn.Embedding(num_nodes, hidden_dim)


        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if conv == "gat":
                # GAT (Velickovic et al. 2018): 4 heads concatenated to hidden_dim, one self-loop per node.
                if hidden_dim % self.GAT_HEADS:
                    raise ValueError(f"hidden_dim {hidden_dim} must be divisible by {self.GAT_HEADS} GAT heads")
                self.convs.append(GATConv(hidden_dim, hidden_dim // self.GAT_HEADS, heads=self.GAT_HEADS))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim) if conv == "sage" else GCNConv(hidden_dim, hidden_dim))








        self.use_dense_adj: bool = False
        self._dense_adj_cache: tuple | None = None

        self._reset_parameters()

    @staticmethod
    def build_dense_mean_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Row-normalised dense adjacency A_hat[i, j] = (#edges j→i) / in_degree(i)"""
        A = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=torch.float32)
        A.index_put_((edge_index[1], edge_index[0]),
                     torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32),
                     accumulate=True)
        deg = A.sum(dim=1, keepdim=True).clamp_(min=1.0)
        return A / deg

    @staticmethod
    def build_dense_gcn_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Symmetric-normalised dense adjacency with self-loops, D^-1/2 (A + I) D^-1/2 (as GCNConv computes)."""
        A = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=torch.float32)
        A.index_put_((edge_index[1], edge_index[0]),
                     torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32),
                     accumulate=True)
        A.fill_diagonal_(1.0)  # one self-loop per node, as add_remaining_self_loops keeps
        d = A.sum(dim=1).pow(-0.5)
        return d[:, None] * A * d[None, :]

    @staticmethod
    def build_dense_edge_counts(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """C[i, j] = #edges j→i, with existing self-loops replaced by exactly one (as GATConv's add_self_loops)."""
        C = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=torch.float32)
        C.index_put_((edge_index[1], edge_index[0]),
                     torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32),
                     accumulate=True)
        C.fill_diagonal_(1.0)
        return C

    @staticmethod
    def dense_gat(conv: "GATConv", h: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """GATConv on a dense edge-count matrix: softmax over each node's in-edges, counting duplicates."""
        H, F_out = conv.heads, conv.out_channels
        x = conv.lin(h).view(-1, H, F_out)                                   # (N, H, F)
        a_src, a_dst = (x * conv.att_src).sum(-1), (x * conv.att_dst).sum(-1)  # (N, H)
        e = F.leaky_relu(a_dst.t()[:, :, None] + a_src.t()[:, None, :], conv.negative_slope)  # (H, N_i, N_j)
        e = e.masked_fill(C[None] == 0, float("-inf"))
        w = torch.exp(e - e.amax(dim=2, keepdim=True)) * C[None]
        alpha = w / w.sum(dim=2, keepdim=True)
        out = torch.einsum("hij,jhf->ihf", alpha, x).reshape(-1, H * F_out)
        return out + conv.bias

    def _get_dense_adj(self, edge_index: torch.Tensor) -> torch.Tensor:
        key = (edge_index.data_ptr(), tuple(edge_index.shape), str(edge_index.device))
        if self._dense_adj_cache is None or self._dense_adj_cache[0] != key:
            build = {"sage": self.build_dense_mean_adj, "gcn": self.build_dense_gcn_adj,
                     "gat": self.build_dense_edge_counts}[self.conv]
            self._dense_adj_cache = (key, build(edge_index, self.num_nodes))
        return self._dense_adj_cache[1]

    def _reset_parameters(self) -> None:
        if self.embedding is not None:
            nn.init.xavier_uniform_(self.embedding.weight)
        if self.feature_proj is not None:
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings via message passing over all nodes"""
        if self.node_features is not None:
            assert self.feature_proj is not None
            h = self.feature_proj(self.node_features)
        else:
            assert self.embedding is not None
            node_ids = torch.arange(self.num_nodes, device=edge_index.device)
            h = self.embedding(node_ids)

        dense = self.use_dense_adj and (self.conv in ("gcn", "gat") or all(hasattr(c, "lin_l") and hasattr(c, "lin_r")
                                                                  for c in self.convs))
        A = self._get_dense_adj(edge_index) if dense else None
        for i, conv in enumerate(self.convs):
            if dense and self.conv == "gat":
                h = self.dense_gat(conv, h, A)
            elif dense and self.conv == "gcn":
                h = A @ conv.lin(h) + conv.bias
            elif dense:

                h = conv.lin_l(A @ h) + conv.lin_r(h)
            else:
                h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)

        return h

    @property
    def uses_fixed_features(self) -> bool:
        """True when operating in Strategy-1 (Feature Bootstrapping) mode."""
        return self.node_features is not None

    def set_feature_delta_(self, delta: torch.Tensor | None) -> None:
        """Apply (or clear) a covariate-drift delta on THIS client's own copy of the fixed feature
        matrix
        """
        if self.node_features is None:
            raise RuntimeError(
                "set_feature_delta_() requires fixed-feature mode "
                "(this encoder was built with node_features=None)."
            )
        if delta is None:
            self.node_features.data.copy_(self._base_node_features)
        else:
            self.node_features.data.copy_(
                self._base_node_features + delta.to(self.node_features.device)
            )
