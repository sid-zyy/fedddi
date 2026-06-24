"""
colab_setup.py
──────────────
One-shot environment validation for Google Colab with PyTorch >= 2.6.

Run once after extracting the zip:

    import os; os.chdir('/content/FedDDI-GNN')
    exec(open('colab_setup.py').read())

What it does
────────────
1. Patches OGB's dataset_pyg.py — fixes torch.load weights_only issue.
   Uses a paren-aware scanner (not regex) to handle nested calls like
   osp.join(...) correctly. Handles all three possible file states:
     • Unpatched   → adds weights_only=False
     • Broken patch → fixes weights_only that ended up inside osp.join args
     • Correct      → leaves unchanged
2. Clears stale cached .pt files so OGB reprocesses cleanly.
3. Verifies all required imports and GPU availability.
4. Verifies all project-level torch.load calls already use weights_only=False.
5. Prints the exact commands to run next.

NOTE: All project-level torch.load calls (trainer, predict, export_embeddings)
already include weights_only=False — no runtime patching of project files needed.
"""

from __future__ import annotations
import importlib
import inspect
import logging
import os
import shutil
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("colab_setup")


# ── Paren-aware torch.load finder (same logic as dataset.py) ─────────────────


def _find_torch_load_calls(src: str) -> List[Tuple[int, int, str]]:
    results: List[Tuple[int, int, str]] = []
    i = 0
    needle = "torch.load("
    while i < len(src):
        idx = src.find(needle, i)
        if idx == -1:
            break
        depth = 0
        j = idx + len(needle)
        start_inner = j
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                if depth == 0:
                    results.append((idx, j + 1, src[start_inner:j]))
                    break
                depth -= 1
            j += 1
        i = idx + 1
    return results


def patch_ogb() -> bool:
    """Patch OGB's dataset_pyg.py to add weights_only=False to torch.load."""
    try:
        import ogb.linkproppred.dataset_pyg as _ogb  # type: ignore
    except ImportError:
        log.error("ogb not installed. Run: pip install ogb")
        return False

    path = inspect.getfile(_ogb)
    log.info("Patching: %s", path)

    with open(path) as f:
        src = f.read()

    original = src

    # Pass 1: fix broken state (weights_only inside nested args)
    calls = _find_torch_load_calls(src)
    for start, end, inner in reversed(calls):
        if ", weights_only=False" in inner:
            clean = inner.replace(", weights_only=False", "")
            src = src[:start] + f"torch.load({clean}, weights_only=False)" + src[end:]

    # Pass 2: add weights_only=False where missing
    calls = _find_torch_load_calls(src)
    for start, end, inner in reversed(calls):
        if "weights_only" not in inner:
            src = src[:start] + f"torch.load({inner}, weights_only=False)" + src[end:]

    if src == original:
        log.info("  Already correctly patched — no changes needed.")
    else:
        with open(path, "w") as f:
            f.write(src)
        count = src.count("weights_only=False")
        log.info("  Patched %d torch.load call(s).", count)

    # Show final state of all torch.load lines for verification
    log.info("  torch.load lines in OGB file:")
    for i, line in enumerate(src.splitlines(), 1):
        if "torch.load" in line:
            log.info("    line %d: %s", i, line.strip())

    return True


def clear_cache(data_root: str = "./data") -> None:
    processed = os.path.join(data_root, "ddi", "processed")
    if os.path.isdir(processed):
        shutil.rmtree(processed)
        log.info("Cleared cache: %s", processed)
    else:
        log.info("No stale cache found (fresh run).")


def check_imports() -> bool:
    pkgs = [
        ("torch", "torch"),
        ("torch_geometric", "torch_geometric"),
        ("ogb", "ogb"),
        ("hydra", "hydra-core"),
        ("omegaconf", "omegaconf"),
        ("wandb", "wandb"),
        ("tqdm", "tqdm"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("networkx", "networkx"),
    ]
    all_ok = True
    for mod, pip_name in pkgs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            log.info("  OK  %-20s %s", mod, ver)
        except ImportError:
            log.error("  MISSING  %-16s  ->  pip install %s", mod, pip_name)
            all_ok = False
    return all_ok


def check_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        log.info("  GPU : %s", torch.cuda.get_device_name(0))
        log.info("  PyTorch : %s  (CUDA %s)", torch.__version__, torch.version.cuda)
    else:
        log.warning("  No GPU detected.")
        log.warning("  Go to: Runtime -> Change runtime type -> T4 GPU")


def verify_project_torch_loads() -> bool:
    """Confirm that all project-level torch.load calls already use weights_only=False.

    These files were patched at source and should not need runtime modification:
      - src/training/trainer.py
      - predict.py
      - scripts/export_embeddings.py
    """
    files_to_check = [
        "src/training/trainer.py",
        "predict.py",
        "scripts/export_embeddings.py",
    ]
    all_ok = True
    for rel_path in files_to_check:
        if not os.path.exists(rel_path):
            log.warning("  MISSING  %s (not found — skipping check)", rel_path)
            continue
        with open(rel_path) as f:
            src = f.read()
        calls = _find_torch_load_calls(src)
        missing = [inner for _, _, inner in calls if "weights_only" not in inner]
        if missing:
            log.error(
                "  FAIL  %s — torch.load without weights_only=False: %s",
                rel_path,
                missing,
            )
            all_ok = False
        else:
            log.info(
                "  OK    %s (%d torch.load call(s), all have weights_only)",
                rel_path,
                len(calls),
            )
    return all_ok


def main() -> None:
    print("\n" + "=" * 62)
    print("  FedDDI Baseline — Colab Environment Setup")
    print("=" * 62)

    print("\n[1/5] Checking imports ...")
    ok = check_imports()
    if not ok:
        print("\nInstall missing packages then re-run.")
        return

    print("\n[2/5] Checking GPU ...")
    check_gpu()

    print("\n[3/5] Patching OGB torch.load ...")
    patch_ogb()

    print("\n[4/5] Clearing stale dataset cache ...")
    clear_cache("./data")

    print("\n[5/5] Verifying project torch.load calls ...")
    if not verify_project_torch_loads():
        log.error(
            "Some project files still have unpatched torch.load calls.\n"
            "This should not happen — please re-download the zip."
        )
        return

    print("\n" + "=" * 62)
    print("  Setup complete! Run these commands next:\n")
    print("  Smoke test (~5 min on T4):")
    print("    !python train.py \\")
    print("        training.num_runs=1 training.epochs=50 \\")
    print("        dataset.num_anchors=200 model.hidden_dim=256 \\")
    print("        wandb.mode=disabled device=cuda")
    print()
    print("  Full leaderboard run (~2-4 hrs on T4):")
    print("    !python train.py \\")
    print("        training.num_runs=10 training.epochs=500 \\")
    print("        dataset.num_anchors=1000 model.hidden_dim=500 \\")
    print("        wandb.mode=disabled device=cuda")
    print("=" * 62 + "\n")


main()
