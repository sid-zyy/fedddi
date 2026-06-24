"""
train.py
────────
Main entry point for the FedDDI centralized baseline.
GraphSAGE encoder + NeuralLinkPredictor (Hadamard MLP decoder).

Usage
─────
  python train.py                                    # Mode A: learned embedding (default)
  python train.py baseline.use_fixed_features=true   # Mode B: fixed features (matches FL)
  python train.py model.hidden_dim=512               # override any param
  python train.py training.num_runs=10               # 10 seeds for leaderboard
  python train.py wandb.mode=disabled                # disable W&B

Baseline modes
──────────────
  A) Learned embedding  (default, baseline.use_fixed_features=false)
     Uses nn.Embedding(num_nodes, hidden_dim). Standard ogbl-ddi leaderboard
     setup; cannot be directly federated (per-client embeddings live in
     orthogonal spaces, so FedAvg aggregation is meaningless).

  B) Fixed global features  (baseline.use_fixed_features=true)
     Uses the SAME compute_fixed_global_features() call as main_fl.py
     (Strategy 1): degree + BFS anchor distances + spectral eigenvectors,
     projected via a shared Linear(feat_dim → hidden_dim). This is the
     MATCHED CENTRALIZED BASELINE: comparing its Hits@20 against the
     federated number isolates the true cost of federation.

  Both modes evaluate using train edges only for message-passing (matching
  ogbl-ddi leaderboard convention).

Paper comparison table:
  | Setting                         | Val Hits@20   |
  |---------------------------------|---------------|
  | Centralized — learned embedding |  TBD          |  (Mode A)
  | Centralized — fixed features    |  TBD          |  (Mode B)
  | Federated   — fixed features    |  TBD          |  (main_fl.py)

Design
──────
For each of the `training.num_runs` independent runs:
  1. Re-seed all RNGs with (base_seed + run_id).
  2. Re-initialise encoder and predictor (fresh random weights).
  3. Run Trainer.train() → returns best val / corresponding test Hits@20.
  4. Log summary to wandb and print results table.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, cast, List

import torch
import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import load_dataset, get_edge_split
from src.data.features import compute_fixed_global_features
from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor
from src.training.trainer import Trainer
from src.utils.utils import get_device, pretty_results, seed_everything, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = get_device(cfg.device)

    data, evaluator = load_dataset(root=cfg.dataset.root)
    split = get_edge_split(root=cfg.dataset.root)

    assert data.edge_index is not None
    assert data.num_nodes is not None

    edge_index = data.edge_index.to(device)
    num_nodes: int = data.num_nodes

    # ── Node feature strategy ─────────────────────────────────────────────────
    # baseline.use_fixed_features=true  →  Mode B: same fixed features as FL
    # baseline.use_fixed_features=false →  Mode A: learned embedding (default)
    baseline_cfg = cfg.get("baseline", {})
    use_fixed_features: bool = bool(baseline_cfg.get("use_fixed_features", False))

    node_features = None

    if use_fixed_features:
        # ── Mode B: fixed global features (matches main_fl.py Strategy 1) ────
        feat_cfg = cfg.dataset.get("fixed_features", {})
        num_anchors_feat  = int(feat_cfg.get("num_anchors",  64))
        num_spectral_feat = int(feat_cfg.get("num_spectral", 32))

        logger.info(
            "Mode B (matched baseline): computing fixed global features "
            "(anchors=%d, spectral=%d) — same as main_fl.py Strategy 1 ...",
            num_anchors_feat, num_spectral_feat,
        )
        node_features = compute_fixed_global_features(
            edge_index.cpu(),
            num_nodes,
            num_anchors=num_anchors_feat,
            num_spectral=num_spectral_feat,
            seed=cfg.training.seed,
        )
        logger.info(
            "Fixed feature dimension: %d → projected to hidden_dim=%d",
            node_features.shape[1], cfg.model.hidden_dim,
        )
        wandb_group = "centralized_fixed_features"
        wandb_tags  = list(cfg.wandb.tags) + ["mode-b-fixed-features", "matched-baseline"]

    elif cfg.dataset.get("use_features", False) and cfg.dataset.get("feature_types", []):
        # ── Legacy path: pre-computed features from features.pt ──────────────
        import os
        feat_path = os.path.join(cfg.dataset.root, "ogbl_ddi", "features.pt")
        if os.path.exists(feat_path):
            feats_dict = torch.load(feat_path, map_location="cpu", weights_only=True)
            feat_list: List[torch.Tensor] = []
            for f_type in cfg.dataset.feature_types:
                if f_type in feats_dict:
                    feat_list.append(feats_dict[f_type])
                else:
                    logger.warning("Feature %s not found in %s", f_type, feat_path)
            if feat_list:
                node_features = torch.cat(feat_list, dim=-1).float()
                logger.info("Loaded node features shape: %s", node_features.shape)
        else:
            logger.warning(
                "Features requested but not found at %s. Please run prepare_features.py first.",
                feat_path,
            )
        wandb_group = "centralized_legacy_features"
        wandb_tags  = list(cfg.wandb.tags) + ["legacy-features"]

    else:
        # ── Mode A: learned embedding (default) ───────────────────────────────
        logger.info(
            "Mode A (learned embedding): using nn.Embedding — "
            "not directly comparable to the federated fixed-feature model. "
            "Run with baseline.use_fixed_features=true for the matched baseline.",
        )
        wandb_group = "centralized_baseline"
        wandb_tags  = list(cfg.wandb.tags)

    all_results: List[Dict[str, float]] = []

    for run_id in range(cfg.training.num_runs):
        run_seed = cfg.training.seed + run_id
        seed_everything(run_seed)
        logger.info(
            "═" * 60 + "\n  RUN %d / %d   (seed=%d)\n" + "═" * 60,
            run_id + 1,
            cfg.training.num_runs,
            run_seed,
        )

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            name=f"run_{run_id}",
            config=cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            tags=wandb_tags,
            group=wandb_group,
            reinit=True,
            mode=cfg.wandb.mode,
        )

        encoder = GraphSAGEEncoder(
            num_nodes=num_nodes,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            normalize=cfg.model.normalize,
            node_features=node_features,
        )

        predictor = NeuralLinkPredictor(
            emb_dim=cfg.model.hidden_dim,
            hidden_dim=cfg.predictor.hidden_dim,
            num_layers=cfg.predictor.num_layers,
            dropout=cfg.predictor.dropout,
            dist_dim=0,  # Distance encoding disabled for centralized baseline
        )

        _log_param_counts(encoder, predictor)

        trainer = Trainer(
            encoder=encoder,
            predictor=predictor,
            edge_index=edge_index,
            split=split,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            run_id=run_id,
            node_dist_enc=None,  # No distance encoding features
        )
        result = trainer.train()
        all_results.append(result)

        wandb.log(
            {
                "run_summary/best_val_hits20": result["best_val_hits20"],
                "run_summary/test_hits20_at_best_val": result[
                    "test_hits20_at_best_val"
                ],
            }
        )
        wandb.finish()

    vals: List[float] = [r["best_val_hits20"] for r in all_results]
    tests: List[float] = [r["test_hits20_at_best_val"] for r in all_results]

    mean_val, std_val = float(np.mean(vals)), float(np.std(vals))
    mean_test, std_test = float(np.mean(tests)), float(np.std(tests))

    logger.info("\n%s", pretty_results(all_results))
    logger.info(
        "\nFinal: Val Hits@20 = %.4f ± %.4f  |  Test Hits@20 = %.4f ± %.4f",
        mean_val,
        std_val,
        mean_test,
        std_test,
    )

    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            name="aggregate_summary",
            config=cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            tags=wandb_tags + ["summary"],
            group=wandb_group,
            reinit=True,
            mode=cfg.wandb.mode,
        )
        wandb.log(
            {
                "aggregate/mean_val_hits20": mean_val,
                "aggregate/std_val_hits20": std_val,
                "aggregate/mean_test_hits20": mean_test,
                "aggregate/std_test_hits20": std_test,
            }
        )
        wandb.finish()


def _log_param_counts(
    encoder: GraphSAGEEncoder, predictor: NeuralLinkPredictor
) -> None:
    enc_p = sum(p.numel() for p in encoder.parameters())
    pred_p = sum(p.numel() for p in predictor.parameters())
    logger.info(
        "Parameter counts — encoder: %d | predictor: %d | total: %d",
        enc_p,
        pred_p,
        enc_p + pred_p,
    )


if __name__ == "__main__":
    main()  # type: ignore
