"""Measures ONE client's real peak GPU memory usage during a full train_epoch, using the exact same
Trainer/model construction path as the real FL run (build_client_trainer)
"""
import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.getcwd())

from src.data.dataset import get_edge_split, load_dataset
from src.data.features import compute_fixed_global_features
from src.training.trainer import build_client_trainer
from src.utils.utils import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


SAFETY_MARGIN = 1.25


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    seed_everything(cfg.training.seed)
    device = get_device(cfg.device)

    if device.type != "cuda":
        print("No CUDA device available — this probe only makes sense on GPU. "
              "Run with device=cuda (and a GPU runtime) to get a real number.")
        return

    torch.cuda.reset_peak_memory_stats(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory


    data, evaluator = load_dataset(root=cfg.dataset.root)
    assert data.edge_index is not None
    assert data.num_nodes is not None
    split = get_edge_split(root=cfg.dataset.root)

    num_nodes = data.num_nodes
    graph = data.edge_index

    feat_cfg = cfg.get("fixed_features", {})
    node_features = compute_fixed_global_features(
        graph.cpu(),
        num_nodes,
        num_anchors=int(feat_cfg.get("num_anchors", 32)),
        num_spectral=int(feat_cfg.get("num_spectral", 16)),
        seed=cfg.training.seed,
    )





    trainer = build_client_trainer(
        cid=0,
        graph=graph,
        node_features=node_features,
        train_edges=split["train"]["edge"],
        valid_split=split["valid"],
        test_split=split["test"],
        num_nodes=num_nodes,
        evaluator=evaluator,
        cfg=cfg,
        device=device,
    )


    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    loss = trainer.train_epoch()

    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mib = peak_bytes / (1024 ** 2)
    total_mib = total_mem / (1024 ** 2)

    max_safe_clients = int(total_mib // (peak_mib * SAFETY_MARGIN))
    max_safe_clients = max(1, max_safe_clients)
    recommended_num_gpus = round(1.0 / max_safe_clients, 3)

    print("\n" + "=" * 72)
    print("VRAM PROBE RESULT (one client, one train_epoch, batch_size="
          f"{cfg.training.batch_size})")
    print("=" * 72)
    print(f"  train_epoch loss (sanity check)      : {loss:.4f}")
    print(f"  peak allocated memory (this client)  : {peak_mib:8.1f} MiB")
    print(f"  total GPU memory available           : {total_mib:8.1f} MiB")
    print(f"  safety margin applied                : {SAFETY_MARGIN}x")
    print(f"  max clients that safely fit together : {max_safe_clients}")
    print(f"  recommended num_gpus for main_fl.py   : {recommended_num_gpus}")
    print("=" * 72)
    print(
        "\nNext step: in main_fl.py, set\n"
        '    "num_gpus": ' + str(recommended_num_gpus) + ",\n"
        "inside backend_config.client_resources (instead of 1.0), OR pass it\n"
        "as a config override if you wire it through Hydra.\n"
    )
    if max_safe_clients < cfg.get("num_clients", 5):
        print(
            f"NOTE: only {max_safe_clients} of {cfg.get('num_clients', 5)} "
            "clients can safely run concurrently on this GPU. Ray will still "
            "run the rest sequentially within each round — you get partial "
            "concurrency, not full 5x. This is still faster than num_gpus=1.0."
        )


if __name__ == "__main__":
    main()
