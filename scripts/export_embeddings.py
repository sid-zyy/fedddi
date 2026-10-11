"""Export trained node embeddings to numpy for downstream analysis"""

from __future__ import annotations

import logging
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import load_dataset
from src.models.encoder import GraphSAGEEncoder
from src.utils.utils import get_device, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)

    checkpoint: str | None = cfg.get("checkpoint", None)
    output: str | None = cfg.get("output", "embeddings.npy")
    if output is None:
        output = "embeddings.npy"

    if checkpoint is None:
        logger.error("Specify: python scripts/export_embeddings.py +checkpoint=PATH")
        sys.exit(1)

    device = get_device(cfg.device)
    data, _ = load_dataset(root=cfg.dataset.root)

    if data.edge_index is None:
        raise ValueError("Dataset missing edge_index.")
    if data.num_nodes is None:
        raise ValueError("Dataset missing num_nodes.")

    edge_index = data.edge_index.to(device)

    encoder = GraphSAGEEncoder(
        num_nodes=data.num_nodes,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        normalize=cfg.model.normalize,
    ).to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()

    with torch.no_grad():
        h = encoder(edge_index).cpu().numpy()

    np.save(output, h)
    logger.info("Saved embeddings %s to '%s'.", h.shape, output)
    print(f"Embeddings shape: {h.shape} → '{output}'")


if __name__ == "__main__":
    main()
