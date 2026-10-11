"""Tests for utility helpers: seeding, device selection, results formatting."""

from __future__ import annotations

import torch
import pytest
import numpy as np

from src.utils.utils import seed_everything, get_device, pretty_results


class TestSeedEverything:
    def test_torch_reproducibility(self):
        seed_everything(42)
        t1 = torch.randn(10)
        seed_everything(42)
        t2 = torch.randn(10)
        torch.testing.assert_close(t1, t2)

    def test_numpy_reproducibility(self):
        seed_everything(7)
        a = np.random.rand(10)
        seed_everything(7)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        seed_everything(0)
        t1 = torch.randn(20)
        seed_everything(1)
        t2 = torch.randn(20)
        assert not torch.allclose(t1, t2)


class TestGetDevice:
    def test_cpu_fallback(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine")
        device = get_device("cuda")
        assert device == torch.device("cpu")


class TestPrettyResults:
    def test_output_is_string(self):
        results = [
            {"best_val_hits20": 0.82, "test_hits20_at_best_val": 0.81},
            {"best_val_hits20": 0.84, "test_hits20_at_best_val": 0.83},
        ]
        table = pretty_results(results)
        assert isinstance(table, str)

    def test_output_contains_mean(self):
        results = [
            {"best_val_hits20": 0.80, "test_hits20_at_best_val": 0.78},
            {"best_val_hits20": 0.84, "test_hits20_at_best_val": 0.82},
        ]
        table = pretty_results(results)
        assert "MEAN" in table

    def test_output_contains_std(self):
        results = [
            {"best_val_hits20": 0.80, "test_hits20_at_best_val": 0.78},
        ]
        table = pretty_results(results)
        assert "STD" in table

    def test_single_run(self):
        results = [{"best_val_hits20": 0.75, "test_hits20_at_best_val": 0.73}]
        table = pretty_results(results)
        assert "0.7500" in table or "0.75" in table

    def test_correct_mean_calculation(self):
        results = [
            {"best_val_hits20": 0.80, "test_hits20_at_best_val": 0.80},
            {"best_val_hits20": 0.90, "test_hits20_at_best_val": 0.90},
        ]
        table = pretty_results(results)

        assert "0.8500" in table

    def test_empty_results(self):
        """Should not crash on empty list."""
        table = pretty_results([])
        assert isinstance(table, str)
