"""Training loop for the GraphSAGE + Hadamard MLP baseline on ogbl-ddi"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch_geometric.utils import negative_sampling  # type: ignore
from ogb.linkproppred import Evaluator  # type: ignore

import wandb

from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor
from src.training.evaluate import evaluate

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the full training lifecycle for one run"""

    def __init__(
        self,
        encoder: GraphSAGEEncoder,
        predictor: NeuralLinkPredictor,
        edge_index: Tensor,
        split: Dict[str, Any],
        evaluator: Evaluator,
        cfg: Any,
        device: torch.device,
        run_id: int = 0,
        node_dist_enc: Tensor | None = None,
    ) -> None:
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.edge_index = edge_index.to(device)
        self.dist_enc = node_dist_enc.to(device) if node_dist_enc is not None else None

        self.split = split
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device
        self.run_id = run_id


        _perf = cfg.get("perf", None) if hasattr(cfg, "get") else None
        _perf = _perf if _perf is not None else {}
        self.encoder.use_dense_adj = bool(_perf.get("dense_adj", False))
        self._fast_neg = bool(_perf.get("fast_negatives", False))
        self._edge_keys: Optional[Tensor] = None

        self.optimizer = Adam(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs
        )

        os.makedirs(cfg.training.save_dir, exist_ok=True)
        self.ckpt_path = os.path.join(cfg.training.save_dir, f"best_run{run_id}.pt")





    def _sample_negatives(self, n: int) -> Tensor:
        """[2, n] node pairs that are not edges of `self.edge_index`"""
        num_nodes = self.encoder.num_nodes
        if not self._fast_neg:
            return negative_sampling(
                edge_index=self.edge_index, num_nodes=num_nodes, num_neg_samples=n
            ).to(self.device)
        if self._edge_keys is None:
            self._edge_keys = torch.unique(self.edge_index[0] * num_nodes + self.edge_index[1])
        keys = self._edge_keys
        us, vs, have = [], [], 0
        while have < n:
            m = int((n - have) * 1.3) + 16
            u = torch.randint(0, num_nodes, (m,), device=self.device)
            v = torch.randint(0, num_nodes, (m,), device=self.device)
            k = u * num_nodes + v
            idx = torch.searchsorted(keys, k).clamp_(max=keys.numel() - 1)
            keep = keys[idx] != k
            us.append(u[keep])
            vs.append(v[keep])
            have += int(keep.sum())
        return torch.stack([torch.cat(us)[:n], torch.cat(vs)[:n]])

    def _aggregated_params(self) -> List[Tensor]:
        """Parameter tensors that participate in FL aggregation"""
        _ENC_KEYS = frozenset(["convs", "feature_proj"])
        _PRED_KEYS = frozenset(["mlp"])
        result: List[Tensor] = []
        for name, p in self.encoder.named_parameters():
            if any(k in name for k in _ENC_KEYS):
                result.append(p)
        for name, p in self.predictor.named_parameters():
            if any(k in name for k in _PRED_KEYS):
                result.append(p)
        return result


    def train_epoch(
        self,
        fedprox_mu: float = 0.0,
        global_params: Optional[List[Tensor]] = None,
    ) -> float:
        """Run one full epoch and return mean binary cross-entropy loss"""
        self.encoder.train()
        self.predictor.train()

        train_pos = self.split["train"]["edge"]
        num_nodes = self.encoder.num_nodes
        batch_size = self.cfg.training.batch_size
        num_neg = self.cfg.training.num_neg_samples
        total_loss = 0.0
        n_batches = 0


        _use_fedprox = fedprox_mu > 0.0 and global_params is not None

        perm = torch.randperm(train_pos.size(0))
        train_pos = train_pos[perm]

        for start in range(0, train_pos.size(0), batch_size):
            end = min(start + batch_size, train_pos.size(0))
            pos_batch = train_pos[start:end].to(self.device)








            h = self.encoder(self.edge_index)

            src_p, dst_p = pos_batch[:, 0], pos_batch[:, 1]


            d_u_p, d_v_p = None, None
            d_u_n, d_v_n = None, None
            if self.dist_enc is not None:
                d_u_p, d_v_p = self.dist_enc[src_p], self.dist_enc[dst_p]

            logits_p = self.predictor(h[src_p], h[dst_p], d_u_p, d_v_p)

            neg_ei = self._sample_negatives(pos_batch.size(0) * num_neg)
            src_n, dst_n = neg_ei[0], neg_ei[1]

            if self.dist_enc is not None:
                d_u_n, d_v_n = self.dist_enc[src_n], self.dist_enc[dst_n]

            logits_n = self.predictor(
                h[src_n], h[dst_n], d_u_n, d_v_n
            )

            loss = F.binary_cross_entropy_with_logits(
                logits_p, torch.ones_like(logits_p)
            ) + F.binary_cross_entropy_with_logits(logits_n, torch.zeros_like(logits_n))













            self.optimizer.zero_grad()
            loss.backward()  # type: ignore






            if self.cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.predictor.parameters()),
                    max_norm=self.cfg.training.grad_clip,
                )







            if _use_fedprox:
                for p, g in zip(
                    self._aggregated_params(),
                    global_params,  # type: ignore[arg-type]
                ):
                    if p.grad is not None:
                        p.grad.data.add_(fedprox_mu * (p.data - g.data))

            self.optimizer.step()  # type: ignore
            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)


    def train(self) -> Dict[str, Any]:
        """Run full training loop with early stopping"""
        cfg = self.cfg
        best_val = 0.0
        test_at_best_val = 0.0
        patience_counter = 0

        history = {
            "eval_epochs": [],
            "val_hits20": [],
            "test_hits20": []
        }

        logger.info(
            "Run %d — starting training for up to %d epochs …",
            self.run_id,
            cfg.training.epochs,
        )

        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.time()
            loss = self.train_epoch()
            elapsed = time.time() - t0

            if epoch % cfg.training.eval_interval == 0:
                val_hits = evaluate(
                    self.encoder,
                    self.predictor,
                    self.edge_index,
                    pos_edges=self.split["valid"]["edge"],
                    neg_edges=self.split["valid"]["edge_neg"],
                    evaluator=self.evaluator,
                    batch_size=cfg.training.batch_size,
                    device=self.device,
                    node_dist_enc=self.dist_enc,
                )
                test_hits = evaluate(
                    self.encoder,
                    self.predictor,
                    self.edge_index,
                    pos_edges=self.split["test"]["edge"],
                    neg_edges=self.split["test"]["edge_neg"],
                    evaluator=self.evaluator,
                    batch_size=cfg.training.batch_size,
                    device=self.device,
                    node_dist_enc=self.dist_enc,
                )

                val_h20 = val_hits.get("Hits@20", 0.0)
                test_h20 = test_hits.get("Hits@20", 0.0)

                history["eval_epochs"].append(epoch)
                history["val_hits20"].append(val_h20)
                history["test_hits20"].append(test_h20)

                wandb.log(
                    {
                        "run": self.run_id,
                        "epoch": epoch,
                        "loss": loss,
                        "val_hits20": val_h20,
                        "test_hits20": test_h20,
                    }
                )

                logger.info(
                    "  epoch %4d | loss %.4f | val Hits@20 %.4f | "
                    "test Hits@20 %.4f | %.1fs",
                    epoch,
                    loss,
                    val_h20,
                    test_h20,
                    elapsed,
                )

                if val_h20 > best_val:
                    best_val = val_h20
                    test_at_best_val = test_h20
                    patience_counter = 0
                    self._save_checkpoint()
                    logger.info(
                        "    ↑ New best val=%.4f, test=%.4f — checkpoint saved.",
                        best_val,
                        test_at_best_val,
                    )
                else:
                    patience_counter += 1

                if patience_counter >= cfg.training.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience %d).",
                        epoch,
                        cfg.training.patience,
                    )
                    break
            else:
                wandb.log({"run": self.run_id, "epoch": epoch, "loss": loss})
                if epoch % 10 == 0:
                    logger.info("  epoch %4d | loss %.4f | %.1fs", epoch, loss, elapsed)

        logger.info(
            "Run %d done — best val Hits@20=%.4f, test Hits@20=%.4f",
            self.run_id,
            best_val,
            test_at_best_val,
        )
        return {
            "best_val_hits20": best_val,
            "test_hits20_at_best_val": test_at_best_val,
            "history": history,
        }


    def _save_checkpoint(self) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
            },
            self.ckpt_path,
        )

    def load_best_checkpoint(self) -> None:
        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(state["encoder"])
        self.predictor.load_state_dict(state["predictor"])
        logger.info("Loaded best checkpoint from '%s'.", self.ckpt_path)


def build_client_trainer(
    cid: int,
    graph: Tensor,
    node_features: Tensor,
    train_edges: Tensor,
    valid_split: Dict[str, Any],
    test_split: Dict[str, Any],
    num_nodes: int,
    evaluator: Evaluator,
    cfg: Any,
    device: torch.device,
    node_dist_enc: Tensor | None = None,
) -> Trainer:
    """Construct one federated client's Trainer"""
    encoder = GraphSAGEEncoder(
        num_nodes=num_nodes,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        normalize=cfg.model.normalize,
        node_features=node_features,
    ).to(device)

    predictor = NeuralLinkPredictor(
        emb_dim=cfg.model.hidden_dim,
        hidden_dim=cfg.predictor.hidden_dim,
        num_layers=cfg.predictor.num_layers,
        dropout=cfg.predictor.dropout,
        dist_dim=cfg.dataset.get("num_anchors", 0),
    ).to(device)

    split = {
        "train": {"edge": train_edges},
        "valid": valid_split,
        "test":  test_split,
    }

    return Trainer(
        encoder=encoder,
        predictor=predictor,
        edge_index=graph,
        split=split,
        evaluator=evaluator,
        cfg=cfg,
        device=device,
        run_id=cid,
        node_dist_enc=node_dist_enc,
    )
