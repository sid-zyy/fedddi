"""
src/utils/utils.py
──────────────────
Shared utility functions: seeding, device selection, logging setup,
results formatting.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops (slight performance cost)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────


def get_device(cfg_device: str = "cuda") -> torch.device:
    """Return the best available device, falling back to CPU if CUDA absent."""
    if cfg_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if cfg_device == "cuda":
            logging.getLogger(__name__).warning(
                "CUDA requested but not available — using CPU."
            )
    logging.getLogger(__name__).info("Using device: %s", device)
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a readable format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Results formatting
# ─────────────────────────────────────────────────────────────────────────────


def pretty_results(results: List[Dict[str, float]]) -> str:
    """Format a list of per-run result dicts into a publication-ready table.

    Example output
    ──────────────
    ┌─────┬────────────────┬─────────────────────────┐
    │ Run │  Val Hits@20   │    Test Hits@20          │
    ├─────┼────────────────┼─────────────────────────┤
    │   0 │         0.8231 │                  0.8105  │
    │   1 │         0.8445 │                  0.8312  │
    …
    │MEAN │         0.8239 │                  0.8239  │
    │ STD │         0.0200 │                  0.0437  │
    └─────┴────────────────┴─────────────────────────┘
    """
    lines = [
        "┌─────┬────────────────┬──────────────────────────┐",
        "│ Run │  Val Hits@20   │  Test Hits@20 (@ best val)│",
        "├─────┼────────────────┼──────────────────────────┤",
    ]
    vals: list[float] = []
    tests: list[float] = []
    for i, r in enumerate(results):
        v = r.get("best_val_hits20", float("nan"))
        t = r.get("test_hits20_at_best_val", float("nan"))
        vals.append(v)
        tests.append(t)
        lines.append(f"│ {i:3d} │       {v:8.4f} │              {t:8.4f}    │")

    lines.append("├─────┼────────────────┼──────────────────────────┤")
    mean_v = float(np.nanmean(vals))
    std_v = float(np.nanstd(vals))
    mean_t = float(np.nanmean(tests))
    std_t = float(np.nanstd(tests))
    lines.append(f"│MEAN │       {mean_v:8.4f} │              {mean_t:8.4f}    │")
    lines.append(f"│ STD │       {std_v:8.4f} │              {std_t:8.4f}    │")
    lines.append("└─────┴────────────────┴──────────────────────────┘")
    return "\n".join(lines)
