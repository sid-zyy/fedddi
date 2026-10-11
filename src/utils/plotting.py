"""Matplotlib plotting utilities for visualizing centralized and federated training runs"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_centralized_run(
    history: Dict[str, List[float]],
    run_id: int,
    save_dir: str = "./plots",
    show: bool = True,
) -> None:
    """Plot Validation and Test Hits@20 for a single centralized training run."""
    eval_epochs = history.get("eval_epochs", [])
    val_hits = history.get("val_hits20", [])
    test_hits = history.get("test_hits20", [])

    if not eval_epochs or not val_hits:
        logger.warning("No history found to plot for run %d.", run_id)
        return

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(
        eval_epochs,
        val_hits,
        label="Validation Hits@20",
        color="#1f77b4",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        eval_epochs,
        test_hits,
        label="Test Hits@20",
        color="#ff7f0e",
        linewidth=2,
        marker="s",
        markersize=4,
    )


    best_idx = int(np.argmax(val_hits))
    best_epoch = eval_epochs[best_idx]
    best_val_val = val_hits[best_idx]
    best_test_val = test_hits[best_idx]

    plt.scatter(
        [best_epoch],
        [best_val_val],
        color="red",
        s=100,
        zorder=5,
        marker="*",
        label=f"Best Val: {best_val_val:.4f} (Ep {best_epoch})",
    )
    plt.scatter(
        [best_epoch],
        [best_test_val],
        color="darkred",
        s=100,
        zorder=5,
        marker="x",
        label=f"Test at Best Val: {best_test_val:.4f}",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Hits@20")
    plt.title(f"Centralized Training - Run {run_id}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"centralized_run_{run_id}.png")
    plt.savefig(save_path, dpi=150)
    logger.info("Saved run %d plot to %s", run_id, save_path)

    if show:
        try:
            plt.show()
        except Exception as e:
            logger.debug("Could not show plot in current environment: %s", e)
    plt.close()


def plot_centralized_summary(
    all_runs_history: List[Dict[str, List[float]]],
    save_dir: str = "./plots",
    show: bool = True,
) -> None:
    """Plot aggregate summary of all centralized training runs."""
    if not all_runs_history:
        return

    eval_epochs = all_runs_history[0].get("eval_epochs", [])
    if not eval_epochs:
        return

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    all_val = []
    all_test = []

    for i, hist in enumerate(all_runs_history):
        val = hist.get("val_hits20", [])
        test = hist.get("test_hits20", [])

        all_val.append(val)
        all_test.append(test)


        hist_epochs = hist.get("eval_epochs", [])
        plt.plot(
            hist_epochs[: len(val)],
            val,
            color="#1f77b4",
            alpha=0.15,
            linestyle="--",
        )
        plt.plot(
            hist_epochs[: len(test)],
            test,
            color="#ff7f0e",
            alpha=0.15,
            linestyle="--",
        )

    min_len = min(len(v) for v in all_val)
    eval_epochs = eval_epochs[:min_len]
    all_val = np.array([v[:min_len] for v in all_val])
    all_test = np.array([t[:min_len] for t in all_test])

    mean_val = np.mean(all_val, axis=0)
    std_val = np.std(all_val, axis=0)
    mean_test = np.mean(all_test, axis=0)
    std_test = np.std(all_test, axis=0)

    plt.plot(
        eval_epochs,
        mean_val,
        label="Mean Val Hits@20",
        color="#1f77b4",
        linewidth=2.5,
    )
    plt.fill_between(
        eval_epochs,
        mean_val - std_val,
        mean_val + std_val,
        color="#1f77b4",
        alpha=0.2,
    )

    plt.plot(
        eval_epochs,
        mean_test,
        label="Mean Test Hits@20",
        color="#ff7f0e",
        linewidth=2.5,
    )
    plt.fill_between(
        eval_epochs,
        mean_test - std_test,
        mean_test + std_test,
        color="#ff7f0e",
        alpha=0.2,
    )


    best_mean_idx = int(np.argmax(mean_val))
    best_epoch = eval_epochs[best_mean_idx]
    best_val_val = mean_val[best_mean_idx]
    best_test_val = mean_test[best_mean_idx]

    plt.scatter(
        [best_epoch],
        [best_val_val],
        color="red",
        s=120,
        zorder=5,
        marker="*",
        label=f"Best Mean Val: {best_val_val:.4f} (Ep {best_epoch})",
    )
    plt.scatter(
        [best_epoch],
        [best_test_val],
        color="darkred",
        s=120,
        zorder=5,
        marker="x",
        label=f"Mean Test at Best Val: {best_test_val:.4f}",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Hits@20")
    plt.title(f"Centralized Training - Aggregate ({len(all_runs_history)} runs)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "centralized_aggregate.png")
    plt.savefig(save_path, dpi=150)
    logger.info("Saved centralized aggregate plot to %s", save_path)

    if show:
        try:
            plt.show()
        except Exception as e:
            logger.debug("Could not show plot in current environment: %s", e)
    plt.close()


def plot_federated_learning(
    history_dict: Dict[str, List[tuple[int, float]]],
    save_dir: str = "./plots",
    show: bool = True,
) -> None:
    """Plot Hits@20 over rounds for Federated Learning from custom history dict."""
    val_history = history_dict.get("val_hits20", [])
    test_history = history_dict.get("test_hits20", [])

    if not val_history:
        logger.warning("No centralized Hits@20 metrics found to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)

    rounds_val = [r for r, _ in val_history]
    val_vals = [float(v) for _, v in val_history]

    plt.figure(figsize=(9, 5))
    plt.plot(
        rounds_val,
        val_vals,
        label="Global Val Hits@20",
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=5,
    )

    if test_history:
        rounds_test = [r for r, _ in test_history]
        test_vals = [float(v) for _, v in test_history]
        plt.plot(
            rounds_test,
            test_vals,
            label="Global Test Hits@20",
            color="#ff7f0e",
            linewidth=2.5,
            marker="s",
            markersize=5,
        )


        best_idx = int(np.argmax(val_vals))
        best_round = rounds_val[best_idx]
        best_val_val = val_vals[best_idx]
        best_test_val = (
            test_vals[best_idx] if best_idx < len(test_vals) else test_vals[-1]
        )

        plt.scatter(
            [best_round],
            [best_val_val],
            color="red",
            s=120,
            zorder=5,
            marker="*",
            label=f"Best Val: {best_val_val:.4f} (Round {best_round})",
        )
        plt.scatter(
            [best_round],
            [best_test_val],
            color="darkred",
            s=120,
            zorder=5,
            marker="x",
            label=f"Test at Best Val: {best_test_val:.4f}",
        )


    dist_val_history = history_dict.get("dist_val_hits20", [])
    dist_local_history = history_dict.get("dist_local_hits20", [])

    if dist_val_history:
        rounds_dist = [r for r, _ in dist_val_history]
        dist_vals = [float(v) for _, v in dist_val_history]
        plt.plot(
            rounds_dist,
            dist_vals,
            label="Aggregated Client Val Hits@20",
            color="#2ca02c",
            linestyle="--",
            alpha=0.7,
        )

    if dist_local_history:
        rounds_local = [r for r, _ in dist_local_history]
        local_vals = [float(v) for _, v in dist_local_history]
        plt.plot(
            rounds_local,
            local_vals,
            label="Aggregated Client Local Hits@20",
            color="#9467bd",
            linestyle=":",
            alpha=0.7,
        )

    plt.xlabel("Federated Round")
    plt.ylabel("Hits@20")
    plt.title("Federated Learning Training Progress")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "federated_learning_history.png")
    plt.savefig(save_path, dpi=150)
    logger.info("Saved federated learning plot to %s", save_path)

    if show:
        try:
            plt.show()
        except Exception as e:
            logger.debug("Could not show plot in current environment: %s", e)
    plt.close()
