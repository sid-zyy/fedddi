"""summarize_variants.py on synthetic run folders: safety columns, exclusion, CCH by group and DRR."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import summarize_variants as sv  # noqa: E402


def _run(folder, name, auroc, mdir, excluded=None, cch=None):
    d = folder / name
    d.mkdir()
    rows = []
    for rnd in range(1, 41):
        r = {"round": rnd, "test_auroc": auroc, "val_auroc": auroc, "test_mdir_high": mdir, "val_mdir_high": mdir}
        if excluded is not None:
            r["excluded"] = json.dumps(excluded)
        if cch is not None:
            r["cch"] = json.dumps(cch)
        rows.append(r)
    pd.DataFrame(rows).to_csv(d / "rounds.csv", index=False)


def test_summary_columns_and_drr(tmp_path):
    _run(tmp_path, "dbrandom-clean_fedavg_seed10_clean_fedavg", 0.95, 0.30)
    _run(tmp_path, "dbrandom-joint-severe_fedavg_seed10_hideHigh_fedavg", 0.94, 0.40,
         excluded=[False] * 10, cch=[5, 7, 1, 1, 0, 0, 0, 0, 0, 0])
    _run(tmp_path, "dbrandom-joint-severe_lfr_seed10_hideHigh_lfr", 0.944, 0.35,
         excluded=[True, False, False, False, True, False, False, False, False, False])
    _run(tmp_path, "dbrandom-joint-severe_fedavg_seed10_hideHigh4_fedavg", 0.935, 0.45)  # not confused with hideHigh
    sv.main(str(tmp_path))
    t = pd.read_csv(tmp_path / "summary_runs.csv").set_index(["attack", "method"])
    assert set(t.index) == {("clean", "fedavg"), ("hideHigh", "fedavg"), ("hideHigh", "lfr"), ("hideHigh4", "fedavg")}
    fa = t.loc[("hideHigh", "fedavg")]
    assert fa["cch_poisoned"] == pytest.approx(6.0) and fa["cch_drifted"] == pytest.approx(1.0)
    assert fa["cch_honest"] == pytest.approx(0.0) and fa["honest_false_exclusion"] == 0.0
    lfr = t.loc[("hideHigh", "lfr")]
    assert lfr["honest_false_exclusion"] == pytest.approx(1 / 6)  # client 4 of 6 honest clients
    # 1 - MDIR: clean 0.70, FedAvg 0.60, LFR 0.65 -> DRR 0.5; AUROC: clean 0.95, FedAvg 0.94, LFR 0.944 -> 0.4
    assert lfr["drr_mdir"] == pytest.approx(0.5)
    assert lfr["drr_auroc"] == pytest.approx(0.4)
    assert fa["drr_mdir"] == pytest.approx(0.0)
