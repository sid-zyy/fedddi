"""
src/models/predictor.py
───────────────────────
NeuralLinkPredictor — Hadamard MLP decoder for link prediction.

Given node embeddings h_u and h_v (each of dimension D):
  1. Forms the element-wise product:  h_{u,v} = h_u ⊙ h_v   [D]
  2. Passes through K MLP layers with ReLU activations:
         h^{(k)} = ReLU( W_k h^{(k-1)} + b_k )    k = 1 … K-1
         h^{(K)} = W_K h^{(K-1)} + b_K
  3. Outputs a single logit (sigmoid applied externally for BCEWithLogitsLoss).

This is the decoder architecture from DLGNN (Li et al., ICML 2021 Workshop),
used here without distance encoding as a strong standalone baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NeuralLinkPredictor(nn.Module):
    """MLP-based link predictor using element-wise product of node embeddings.

    Parameters
    ----------
    emb_dim    : dimension of GNN node embeddings (= model.hidden_dim)
    hidden_dim : hidden dimension for the MLP layers
    num_layers : total number of linear layers (including output layer)
    dropout    : dropout applied between hidden layers
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        dist_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dist_dim = dist_dim

        layers: list[nn.Module] = []
        current_dim = emb_dim + 3 * dist_dim
        for k in range(num_layers):
            out_dim = 1 if k == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(current_dim, out_dim))
            if k < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        h_u: Tensor,
        h_v: Tensor,
        d_u: Tensor | None = None,
        d_v: Tensor | None = None,
    ) -> Tensor:
        """Compute link prediction logits for a batch of node pairs.

        Parameters
        ----------
        h_u : FloatTensor [B, emb_dim]  — source node embeddings
        h_v : FloatTensor [B, emb_dim]  — destination node embeddings
        d_u : FloatTensor [B, dist_dim] — source distance encodings (optional)
        d_v : FloatTensor [B, dist_dim] — destination distance encodings (optional)

        Returns
        -------
        logits : FloatTensor [B, 1]
        """
        h_uv = h_u * h_v  # element-wise product [B, emb_dim]

        if self.dist_dim > 0 and d_u is not None and d_v is not None:
            # compute min, sum, diff
            d_min = torch.minimum(d_u, d_v)
            d_sum = d_u + d_v
            d_diff = torch.abs(d_u - d_v)
            dist_feat = torch.cat([d_min, d_sum, d_diff], dim=-1)
            h_uv = torch.cat([h_uv, dist_feat], dim=-1)

        return self.mlp(h_uv)  # [B, 1]
