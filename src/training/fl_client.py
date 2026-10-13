"""Flower NumPyClient for FedDDI-GNN"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import Scalar

from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


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
        algorithm: str = "fedavg",
        mu: float = 0.0,
        num_rounds: int = 1,
        update_scale: float = 1.0,
        poison_target_edges: Optional[torch.Tensor] = None,
        drifted_node_mask: Optional[torch.Tensor] = None,
        c_id_int: Optional[int] = None,
        poisoning_simulator: Optional[object] = None,
        data_drift_simulator: Optional[object] = None,
    ):
        self.cid = cid
        self.trainer = trainer
        self.local_epochs = local_epochs
        self.device = device
        self.algorithm = algorithm.lower()
        self.mu = mu
        self.num_rounds = num_rounds






























        self.update_scale = update_scale
        self.poison_target_edges = poison_target_edges
        # Fabricated (emerged) and suppressed (false-negative) edges move in opposite directions under
        # a successful attack, so they are also scored separately.
        self.poison_fabricated_edges: Optional[torch.Tensor] = None
        self.poison_suppressed_edges: Optional[torch.Tensor] = None
        self.c_id_int = c_id_int
        self.poisoning_simulator = poisoning_simulator
        self.data_drift_simulator = data_drift_simulator







        self.drifted_node_mask = drifted_node_mask

        self.encoder = trainer.encoder
        self.predictor = trainer.predictor


        self._num_model_params: int = len(self.get_parameters(config={}))



        if not self.encoder.uses_fixed_features:
            logger.warning(
                "[Client %s] Encoder is in LEGACY EMBEDDING mode (no fixed features). "
                "Federated Hits@20 will be near-zero due to orthogonal client spaces. "
                "Set use_features=true and feature_types=[structural] to enable "
                "Feature Bootstrapping (fixed global features).",
                cid,
            )


    def _refresh_round_state(self, config: Dict[str, Scalar]) -> None:
        """Re-derive this round's poisoned dataset, update_scale, poison target edges, and drift
        delta from `config["server_round"]`
        """
        server_round = int(config.get("server_round", 0))

        if self.poisoning_simulator is not None and self.c_id_int is not None:
            current_datasets = self.poisoning_simulator.step(server_round)
            ds = current_datasets[self.c_id_int]
            self.trainer.split["train"]["edge"] = ds.edges.to(self.device)
            self.poison_fabricated_edges = ds.emerged_edges if ds.emerged_edges.numel() else None
            self.poison_suppressed_edges = ds.false_negative_edges if ds.false_negative_edges.numel() else None
            if ds.emerged_edges.numel() or ds.false_negative_edges.numel():
                self.poison_target_edges = torch.cat(
                    [ds.emerged_edges, ds.false_negative_edges], dim=0
                ).to(self.device)
            else:
                self.poison_target_edges = None
            self.update_scale = self.poisoning_simulator.update_scale(
                self.c_id_int, server_round
            )

        if self.data_drift_simulator is not None and self.c_id_int is not None:
            deltas = self.data_drift_simulator.step(server_round)
            self.encoder.set_feature_delta_(deltas.get(self.c_id_int))
            self.drifted_node_mask = self.data_drift_simulator.drifted_node_mask(
                self.c_id_int
            )


    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return the subset of parameters to be aggregated by the server"""
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


    def _sync_cuda(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Install global weights, run local_epochs of SGD, return updated weights"""


        _t_fit0 = time.perf_counter()
        self._refresh_round_state(config)


        for group in self.trainer.optimizer.param_groups:
            group["params_backup_lr"] = group["lr"]
        self.trainer.optimizer.state.clear()
        for group in self.trainer.optimizer.param_groups:
            group["lr"] = group.get("params_backup_lr", self.trainer.cfg.training.lr)


        self.set_parameters(parameters)




        global_params_t: Optional[List[torch.Tensor]] = None
        if self.algorithm == "fedprox" and self.mu > 0.0:
            global_params_t = [
                torch.tensor(p, dtype=torch.float32, device=self.device)
                for p in parameters
            ]

        logger.info(
            "[Client %s] Starting local training (%s) for %d epochs",
            self.cid,
            self.algorithm,
            self.local_epochs,
        )


        for param_group in self.trainer.optimizer.param_groups:
            param_group["lr"] = self.trainer.cfg.training.lr

        self.trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.trainer.optimizer, T_max=max(self.num_rounds * self.local_epochs, 1)
        )

        original_epochs = self.trainer.cfg.training.epochs
        self.trainer.cfg.training.epochs = self.local_epochs

        self._sync_cuda()
        _t_train0 = time.perf_counter()
        loss = 0.0
        for _ in range(self.local_epochs):
            loss = self.trainer.train_epoch(
                fedprox_mu=self.mu if self.algorithm == "fedprox" else 0.0,
                global_params=global_params_t,
            )

        self.trainer.cfg.training.epochs = original_epochs
        self._sync_cuda()
        _t_train = time.perf_counter() - _t_train0

        print(
            f"[timing] client {self.cid} round {int(config.get('server_round', -1))} fit: "
            f"setup {_t_train0 - _t_fit0:.2f}s train {_t_train:.2f}s "
            f"(epochs={self.local_epochs} dense_adj={getattr(self.trainer.encoder, 'use_dense_adj', '?')} "
            f"fast_neg={getattr(self.trainer, '_fast_neg', '?')})",
            flush=True,
        )

        new_params = self.get_parameters(config={})






        if self.update_scale != 1.0:
            new_params = [
                old + self.update_scale * (new - old)
                for old, new in zip(parameters, new_params)
            ]
            logger.info(
                "[Client %s] Malicious client — boosting returned update "
                "by update_scale=%.2f before aggregation.",
                self.cid,
                self.update_scale,
            )

        num_examples = int(self.trainer.edge_index.size(1))
        result = (
            new_params,
            num_examples,
            {"loss": float(loss), "cid": int(self.cid)},
        )

















        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result


    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Score the global model on the global validation set"""
        self._refresh_round_state(config)
        self.set_parameters(parameters)

        from src.training.evaluate import evaluate as eval_fn
        from src.training.evaluate import score_edge_set

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






        if self.drifted_node_mask is not None:
            mask = self.drifted_node_mask.to(self.device)
            incident = mask[val_edges[:, 0]] | mask[val_edges[:, 1]]
            drift_region_edges = val_edges[incident]
            if drift_region_edges.size(0) > 0:
                drift_hits = eval_fn(
                    encoder=self.encoder,
                    predictor=self.predictor,
                    edge_index=self.trainer.edge_index,
                    pos_edges=drift_region_edges,
                    neg_edges=val_edges_neg,
                    evaluator=self.trainer.evaluator,
                    batch_size=self.trainer.cfg.training.batch_size,
                    device=self.device,
                    node_dist_enc=self.trainer.dist_enc,
                )
                metrics["drift_region_Hits@20"] = float(drift_hits.get("Hits@20", 0.0))
                metrics["drift_region_n"] = int(drift_region_edges.size(0))
                logger.info(
                    "[Client %s] evaluate — drift-region Val Hits@20: %.4f (n=%d)",
                    self.cid,
                    metrics["drift_region_Hits@20"],
                    drift_region_edges.size(0),
                )









            non_drift_edges = val_edges[~incident]
            if non_drift_edges.size(0) > 0:
                non_drift_hits = eval_fn(
                    encoder=self.encoder,
                    predictor=self.predictor,
                    edge_index=self.trainer.edge_index,
                    pos_edges=non_drift_edges,
                    neg_edges=val_edges_neg,
                    evaluator=self.trainer.evaluator,
                    batch_size=self.trainer.cfg.training.batch_size,
                    device=self.device,
                    node_dist_enc=self.trainer.dist_enc,
                )
                metrics["non_drift_region_Hits@20"] = float(non_drift_hits.get("Hits@20", 0.0))
                metrics["non_drift_region_n"] = int(non_drift_edges.size(0))




        if self.poison_target_edges is not None and self.poison_target_edges.numel() > 0:
            attack_scores = score_edge_set(
                encoder=self.encoder,
                predictor=self.predictor,
                edge_index=self.trainer.edge_index,
                edges=self.poison_target_edges.to(self.device),
                batch_size=self.trainer.cfg.training.batch_size,
                device=self.device,
                node_dist_enc=self.trainer.dist_enc,
            )
            metrics["poison_attack_mean_prob"] = attack_scores["mean_prob"]
            metrics["poison_attack_n"] = attack_scores["n"]
            logger.info(
                "[Client %s] evaluate — poisoning attack mean_prob: %.4f (n=%d)",
                self.cid,
                attack_scores["mean_prob"],
                attack_scores["n"],
            )

        for key, edges in (("poison_fabricated_mean_prob", self.poison_fabricated_edges),
                           ("poison_suppressed_mean_prob", self.poison_suppressed_edges)):
            if edges is not None and edges.numel() > 0:
                metrics[key] = score_edge_set(
                    encoder=self.encoder,
                    predictor=self.predictor,
                    edge_index=self.trainer.edge_index,
                    edges=edges.to(self.device),
                    batch_size=self.trainer.cfg.training.batch_size,
                    device=self.device,
                    node_dist_enc=self.trainer.dist_enc,
                )["mean_prob"]

        num_examples = int(val_edges.size(0))




        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return 0.0, num_examples, metrics
