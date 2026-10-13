"""Learning-rate decay across federated rounds (FedDDIClient round_lr_decay)."""
from __future__ import annotations

import pytest
import torch

from src.training.fl_client import round_lr_factor


def test_factor_runs_from_one_to_floor():
    assert round_lr_factor(1, 0, 5, 100) == pytest.approx(1.0)
    assert round_lr_factor(100, 5, 5, 100) == pytest.approx(0.05)
    assert round_lr_factor(51, 0, 5, 100) == pytest.approx(0.05 + 0.95 * 0.5)


def test_factor_continues_across_rounds_and_never_rises():
    vals = [round_lr_factor(r, k, 2, 10) for r in range(1, 11) for k in range(2)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))
    # Round r+1 starts where round r ended.
    assert round_lr_factor(4, 0, 2, 10) == pytest.approx(round_lr_factor(3, 2, 2, 10))


def test_client_reset_sequence_decays_instead_of_restarting():
    """Replays what FedDDIClient.fit does each round: reset lr to the base value, build the
    scheduler, then step it once per local epoch."""
    base, local_epochs, rounds = 0.01, 2, 5
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w], lr=base)
    seen = []
    for rnd in range(1, rounds + 1):
        for g in opt.param_groups:
            g["lr"] = base
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k, rnd=rnd: round_lr_factor(rnd, k, local_epochs, rounds))
        for _ in range(local_epochs):
            seen.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
    assert seen[0] == pytest.approx(base)
    assert all(b < a for a, b in zip(seen, seen[1:]))
    assert seen[-1] < 0.2 * base
