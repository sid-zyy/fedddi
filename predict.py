"""Inference script: load a saved checkpoint and score arbitrary node pairs"""

from __future__ import annotations

import logging
import sys

import hydra
import torch
from omegaconf import DictConfig

from typing import Dict, Any
from src.data.dataset import load_dataset, get_edge_split
from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor
from src.training.evaluate import evaluate, score_edges
from src.utils.utils import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)

    checkpoint: str | None = cfg.get("checkpoint", None)
    input_csv: str | None = cfg.get("input_csv", None)
    output_csv: str | None = cfg.get("output_csv", "scores.csv")
    if output_csv is None:
        output_csv = "scores.csv"
    mode: str = cfg.get("mode", "test_split")

    if checkpoint is None:
        logger.error("No checkpoint specified. Use: python predict.py +checkpoint=PATH")
        sys.exit(1)

    device = get_device(cfg.device)
    seed_everything(cfg.training.seed)

    data, evaluator = load_dataset(root=cfg.dataset.root)
    split = get_edge_split(root=cfg.dataset.root)
    assert data.edge_index is not None
    assert data.num_nodes is not None
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes

    encoder = GraphSAGEEncoder(
        num_nodes=num_nodes,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        normalize=cfg.model.normalize,
    ).to(device)

    predictor = NeuralLinkPredictor(
        emb_dim=cfg.model.hidden_dim,
        hidden_dim=cfg.predictor.hidden_dim,
        num_layers=cfg.predictor.num_layers,
        dropout=cfg.predictor.dropout,
    ).to(device)

    logger.info("Loading checkpoint from '%s' …", checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    logger.info("Checkpoint loaded.")

    encoder.eval()
    predictor.eval()

    if mode == "test_split":
        _eval_test_split(encoder, predictor, edge_index, split, evaluator, cfg, device)
    elif mode == "csv":
        if input_csv is None:
            logger.error("mode=csv requires input_csv=PATH")
            sys.exit(1)
        _score_csv(encoder, predictor, edge_index, input_csv, output_csv, cfg, device)
    else:
        logger.error("Unknown mode '%s'. Use: test_split | csv", mode)
        sys.exit(1)


def _eval_test_split(
    encoder: GraphSAGEEncoder,
    predictor: NeuralLinkPredictor,
    edge_index: torch.Tensor,
    split: Dict[str, Any],
    evaluator: Any,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    val_hits = evaluate(
        encoder,
        predictor,
        edge_index,
        pos_edges=split["valid"]["edge"],
        neg_edges=split["valid"]["edge_neg"],
        evaluator=evaluator,
        batch_size=cfg.training.batch_size,
        device=device,
    )
    test_hits = evaluate(
        encoder,
        predictor,
        edge_index,
        pos_edges=split["test"]["edge"],
        neg_edges=split["test"]["edge_neg"],
        evaluator=evaluator,
        batch_size=cfg.training.batch_size,
        device=device,
    )
    print("\n── Evaluation Results ──────────────────────────")
    for k, v in val_hits.items():
        print(f"  Val  {k}: {v:.4f}")
    for k, v in test_hits.items():
        print(f"  Test {k}: {v:.4f}")
    print("────────────────────────────────────────────────\n")


def _score_csv(
    encoder: GraphSAGEEncoder,
    predictor: NeuralLinkPredictor,
    edge_index: torch.Tensor,
    input_csv: str,
    output_csv: str,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    import csv

    logger.info("Reading pairs from '%s' …", input_csv)
    srcs: list[int] = []
    dsts: list[int] = []
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            srcs.append(int(row["src"]))
            dsts.append(int(row["dst"]))

    pairs = torch.tensor(list(zip(srcs, dsts)), dtype=torch.long)
    logger.info("Scoring %d node pairs …", pairs.size(0))

    with torch.no_grad():
        h = encoder(edge_index.to(device))
        scores = score_edges(h, predictor, pairs, cfg.training.batch_size, device)

    probs = torch.sigmoid(scores).numpy()
    scores_np = scores.numpy()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "score", "prob"])
        for src, dst, sc, pr in zip(srcs, dsts, scores_np.tolist(), probs.tolist()):
            writer.writerow([src, dst, f"{sc:.6f}", f"{pr:.6f}"])

    print(f"\nScored {len(srcs)} pairs → '{output_csv}'")
    print(f"  Score range: [{scores_np.min():.3f}, {scores_np.max():.3f}]")
    print(f"  Prob  range: [{probs.min():.3f},  {probs.max():.3f}]")


if __name__ == "__main__":
    main()  # type: ignore
