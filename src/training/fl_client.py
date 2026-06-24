"""
src/training/fl_client.py  (Strategy 1 — Feature Bootstrapping)
─────────────────────────────────────────────────────────────────
Flower NumPyClient for FedDDI-GNN.

Key design points
──────────────────
1. get_parameters / set_parameters include `feature_proj` unconditionally.
   In Strategy 1 every client has a feature_proj (there is no private
   embedding table), so this is always correct. In the legacy embedding mode
   the parameter filter still excludes `embedding` as before.

2. fit() resets the optimizer state at the start of each FL round so that
   stale momentum from the previous round does not corrupt the update
   direction after new server weights are installed.

3. evaluate() reports federated_Hits@20 — the GLOBAL validation set scored
   with this client's own encoder. This is meaningful because all clients
   share the same fixed feature coordinate frame (Strategy 1) AND the same
   global graph (every client trains and evaluates over the complete,
   public DDI edge_index — there is no client-specific subgraph to score
   "locally" against; see main_fl.py's "Graph ownership model" docstring).

4. The `server_diagnostic` path is GONE from the client. The server has its
   own global encoder for diagnostics (main_fl.py). The client only needs to
   report the metric above.

Aggregation correctness (why evaluate() gives real numbers)
───────────────────────────────────────────────────────────────
OLD (broken): each client's embedding lived in a private Rⁿ. Aggregating
  convs/MLP on top of orthogonal Rⁿ spaces produced a "global model" whose
  node representations had no consistent meaning: encoder(v) on client A vs.
  encoder(v) on client B pointed in unrelated directions. The predictor MLP,
  trying to make sense of Hadamard products of such vectors, produced random
  logits — hence Hits@20 ≈ 0.

NEW (Strategy 1 + shared global graph): every client computes
  h = feature_proj(fixed_feats) where fixed_feats is identical across all
  clients, AND the SAGEConv message-passing graph itself is the same
  complete global edge_index on every client. FedAvg averages the
  gradients of feature_proj → produces a *consensus projection* that maps
  the shared feature space into hidden_dim. SAGEConv and MLP are
  aggregated on top of this aligned input and trained against the true
  global neighborhood structure, so the aggregated model is
  self-consistent: evaluating it on ANY client with the full edge
  neighbourhood produces logits in the same score space as every other
  client — and against the global validation set, that means real Hits@20
  that should approach the centralized baseline, not just a non-degenerate
  one.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import Scalar

from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

# Parameter name groups — kept as module-level constants so get/set_parameters
# and main_fl.py's initial_parameters block are always in sync.
# ALL of these are included in the aggregation; only `embedding` is excluded.
_ENCODER_AGG_KEYS = frozenset(["convs", "feature_proj"])
_PREDICTOR_AGG_KEYS = frozenset(["mlp"])


def _is_aggregated_encoder_param(name: str) -> bool:
    return any(key in name for key in _ENCODER_AGG_KEYS)


def _is_aggregated_predictor_param(name: str) -> bool:
    return any(key in name for key in _PREDICTOR_AGG_KEYS)


class FedDDIClient(NumPyClient):
    def __init__(
        self,
        cid: str,
        trainer: Trainer,
        local_epochs: int,
        device: torch.device,
    ):
        self.cid = cid
        self.trainer = trainer
        self.local_epochs = local_epochs
        self.device = device

        self.encoder = trainer.encoder
        self.predictor = trainer.predictor

        # Warn loudly if running in legacy embedding mode — federated results
        # will be near-zero regardless of training budget.
        if not self.encoder.uses_fixed_features:
            logger.warning(
                "[Client %s] Encoder is in LEGACY EMBEDDING mode (no fixed features). "
                "Federated Hits@20 will be near-zero due to orthogonal client spaces. "
                "Set use_features=true and feature_types=[structural] to enable "
                "Strategy 1 (Feature Bootstrapping).",
                cid,
            )

    # ── parameter serialisation ───────────────────────────────────────────────

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return the subset of parameters to be aggregated by the server.

        Includes: encoder convs, encoder feature_proj (Strategy 1), predictor MLP.
        Excludes: encoder embedding (legacy private table — should not exist in
                  Strategy 1, but guarded against for backward compatibility).
        """
        agg_params: List[np.ndarray] = []
        for name, param in self.encoder.named_parameters():
            if _is_aggregated_encoder_param(name):
                agg_params.append(param.cpu().detach().numpy())

        for name, param in self.predictor.named_parameters():
            if _is_aggregated_predictor_param(name):
                agg_params.append(param.cpu().detach().numpy())

        return agg_params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Install aggregated parameters received from the server."""
        idx = 0
        state_dict = self.encoder.state_dict()
        for name, _ in self.encoder.named_parameters():
            if _is_aggregated_encoder_param(name):
                state_dict[name] = torch.tensor(
                    parameters[idx], dtype=torch.float32
                ).to(self.device)
                idx += 1
        self.encoder.load_state_dict(state_dict, strict=False)

        state_dict_pred = self.predictor.state_dict()
        for name, _ in self.predictor.named_parameters():
            if _is_aggregated_predictor_param(name):
                state_dict_pred[name] = torch.tensor(
                    parameters[idx], dtype=torch.float32
                ).to(self.device)
                idx += 1
        self.predictor.load_state_dict(state_dict_pred, strict=False)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Install global weights, run local_epochs of SGD, return updated weights.

        Critical fix: optimizer state is reset before installing the new server
        weights. Without this, Adam's first-moment / second-moment estimates
        from round N are applied to the gradient of round N+1's starting point,
        which was set by a *different* aggregation of all clients. Stale
        momentum corrupts the early steps of each local training phase and
        slows convergence — especially noticeable with non-IID data where the
        aggregated model may be far from the local optimum.
        """
        # Reset optimizer: clear stale momentum from the previous round.
        for group in self.trainer.optimizer.param_groups:
            group["params_backup_lr"] = group["lr"]  # save LR
        self.trainer.optimizer.state.clear()
        for group in self.trainer.optimizer.param_groups:
            group["lr"] = group.get("params_backup_lr", self.trainer.cfg.training.lr)

        # Install the freshly aggregated global model.
        self.set_parameters(parameters)

        logger.info(
            "[Client %s] Starting local training for %d epochs",
            self.cid,
            self.local_epochs,
        )

        # Reset LR to initial and rebuild cosine schedule for this round.
        for param_group in self.trainer.optimizer.param_groups:
            param_group["lr"] = self.trainer.cfg.training.lr

        self.trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.trainer.optimizer, T_max=max(self.local_epochs, 1)
        )

        original_epochs = self.trainer.cfg.training.epochs
        self.trainer.cfg.training.epochs = self.local_epochs

        loss = 0.0
        for _ in range(self.local_epochs):
            loss = self.trainer.train_epoch()

        self.trainer.cfg.training.epochs = original_epochs

        num_examples = int(self.trainer.edge_index.size(1))
        return (
            self.get_parameters(config={}),
            num_examples,
            {"loss": float(loss), "cid": int(self.cid)},
        )

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Score the global model on the global validation set.

        This uses THIS client's encoder — which is correct in Strategy 1
        because the encoder's input (fixed_features → feature_proj) is
        globally aligned, and the message-passing graph is the same shared
        global edge_index used by every client. The aggregated feature_proj
        + convs + MLP produce a model that scores edges in a globally
        consistent space.

        Returns
        -------
        (loss, num_examples, metrics)
          metrics["Hits@20"] — federated (global val set, this client's encoder)
        """
        self.set_parameters(parameters)

        from src.training.evaluate import evaluate as eval_fn

        val_edges     = self.trainer.split["valid"]["edge"].to(self.device)
        val_edges_neg = self.trainer.split["valid"]["edge_neg"].to(self.device)

        global_val_hits = eval_fn(
            encoder=self.encoder,
            predictor=self.predictor,
            edge_index=self.trainer.edge_index,
            pos_edges=val_edges,
            neg_edges=val_edges_neg,
            evaluator=self.trainer.evaluator,
            batch_size=self.trainer.cfg.training.batch_size,
            device=self.device,
            node_dist_enc=self.trainer.dist_enc,
        )
        global_val_h20 = float(global_val_hits.get("Hits@20", 0.0))

        metrics: Dict[str, Scalar] = {"Hits@20": global_val_h20}

        logger.info(
            "[Client %s] evaluate — federated Val Hits@20: %.4f",
            self.cid,
            global_val_h20,
        )

        num_examples = int(val_edges.size(0))
        return 0.0, num_examples, metrics
