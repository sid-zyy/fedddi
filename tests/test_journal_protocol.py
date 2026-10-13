"""Torch-free tests for scripts/journal_protocol.py using synthetic result CSVs."""
import json, os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import journal_protocol as jp


def test_specs_are_disjoint_and_valid():
    p, d = jp.build_specs()
    pc = {e["client_id"] for e in p["p10"]}; dc = {e["client_id"] for e in d["d10"]}
    assert pc == {0, 1} and dc == {2, 3} and not (pc & dc)
    assert all(e["start_round"] >= jp.PROTOCOL["attack_start"] for e in p["p10"] + d["d10"])
    assert max(pc | dc) < jp.PROTOCOL["num_clients"]


def test_run_order_finishes_tier1_for_all_seeds_first():
    runs = jp.make_runs()
    assert len(runs) == 5 * (2 * 2) + 5 * (2 * 1) + 5 * (2 * 3) == 60
    tiers = [r["tier"] for r in runs]
    assert tiers == sorted(tiers)


def test_overrides_contain_protocol_and_perf():
    r = [x for x in jp.make_runs() if x["method"] == "caga" and x["cond"] == "joint"][0]
    ov = jp.overrides_for(r, "/tmp/x")
    s = " ".join(ov)
    assert "num_clients=10" in s and "num_rounds=100" in s and "local_epochs=5" in s
    assert "observations_per_client=200000" in s and "defense.mode=caga" in s
    assert "perf.dense_adj=true" in s and "defense.root_device=cuda" in s and "perf.actor_cache=true" in s
    assert "poisoning.events=" in s and "data_drift.events=" in s
    ov0 = jp.overrides_for(r, "/tmp/x", use_perf=False)
    assert not any("perf." in o or "root_device" in o for o in ov0)
    clean = [x for x in jp.make_runs() if x["cond"] == "clean" and x["method"] == "fedavg"][0]
    sc = " ".join(jp.overrides_for(clean, "/tmp/x"))
    assert "poisoning.events" not in sc and "data_drift.events" not in sc and "defense.mode" not in sc


def _fake_results(root, rounds=100, truncate=None):
    rng = np.random.default_rng(0)
    for r in jp.make_runs():
        d = os.path.join(root, r["name"]); os.makedirs(d)
        n = rounds if r["name"] != truncate else 50
        base = {"fedavg": 0.30, "normclip": 0.30, "caga": 0.31}[r["method"]]
        drop = 0.0 if r["cond"] == "clean" else (0.00 if r["method"] == "caga" else 0.06)
        rows = []
        for k in range(n + 1):
            v = base - (drop if k >= 50 else 0) + rng.normal(0, 0.01)
            row = dict(round=k, test_hits20=v, val_hits20=v, poison_attack_mean_prob=0.1, drift_region_hits20=v)
            if r["method"] == "caga" and r["cond"] == "joint":
                verd = ["rejected", "scaled", "phi_gated", "phi_gated"] + ["accepted"] * 6
                row.update(caga_client_keys=json.dumps([str(i) for i in range(10)]), caga_verdict=json.dumps(verd),
                           caga_scale_phi=json.dumps([1, 1, 0, 0.5] + [1] * 6))
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(d, f"results_fedavg_{r['name']}.csv"), index=False)


def test_resume_windows_pairing_and_routing():
    with tempfile.TemporaryDirectory() as t:
        _fake_results(t, truncate="joint_fedavg_seed3")
        assert not jp.is_complete(os.path.join(t, "joint_fedavg_seed3"), 100)
        assert jp.is_complete(os.path.join(t, "joint_fedavg_seed0"), 100)
        tbl = jp.window_table(t)
        assert len(tbl) == 59
        s = jp.summarize(tbl)
        assert abs(s.loc[("joint", "fedavg"), "mean"] - 0.24) < 0.02
        pv = jp.paired_vs_baseline(tbl)
        j = pv[(pv.cond == "joint") & (pv.method == "caga")].iloc[0]
        assert j.mean_diff > 0.05 and j.excludes_zero and j.n_seeds == 4
        c = pv[(pv.cond == "clean") & (pv.method == "caga")].iloc[0]
        assert 0.0 < c.mean_diff < 0.03
        rt = jp.caga_routing(t)
        assert rt.loc["honest", "flagged"] == 0.0 and rt.loc["poisoned", "flagged"] == 1.0
        assert rt.loc["drifted", "phi_gated"] == 1.0


def test_live_watch_prints_each_round_once(tmp_path, capsys):
    """_watch streams one line per round while a (fake) run grows its per-round CSV."""
    import subprocess, textwrap
    run_dir = str(tmp_path / "joint_caga_seed0"); os.makedirs(run_dir)
    script = textwrap.dedent(f"""
        import csv, time
        path = r"{run_dir}/results_x_seed0.csv"
        cols = ["round","train_loss","val_hits20","test_hits20","federated_val_hits20",
                "poison_attack_mean_prob","drift_region_hits20","caga_client_keys","caga_verdict"]
        rows = {{}}
        def flush():
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
                for r in sorted(rows): w.writerow({{c: rows[r].get(c, "") for c in cols}})
        rows[0] = {{"round": 0, "val_hits20": .01, "test_hits20": .01}}; flush(); time.sleep(.3)
        for r in (1, 2, 3):
            rows[r] = {{"round": r, "train_loss": .5}}; flush()
            rows[r].update(val_hits20=.1*r, test_hits20=.1*r,
                           caga_client_keys='["0","1","2"]', caga_verdict="['rejected','accepted','phi_gated']"); flush()
            time.sleep(.3)
            rows[r].update(federated_val_hits20=.1*r, poison_attack_mean_prob=.2, drift_region_hits20=.3); flush()
            time.sleep(.3)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script])
    jp._watch(proc, run_dir, total_rounds=3, attack_start=2, poll_s=0.1, name="fake")
    out = capsys.readouterr().out.splitlines()
    lines = [l for l in out if l.startswith("r0")]
    assert [l[:4] for l in lines] == ["r000", "r001", "r002", "r003"]
    assert "ATK" in lines[2] and "ATK" not in lines[1]
    assert "poison_p 0.2000" in lines[1] and "drift_h20 0.3000" in lines[1]
    assert "caga flagged [0,2]" in lines[1] and "ETA" in lines[3]


def test_phase_breakdown_separates_training_from_overhead(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        "[2026-09-21 06:00:00,000][flwr][INFO] - [ROUND 1]",
        "(ClientAppActor pid=1) [timing] client 0 round 1 fit: setup 0.50s train 6.00s (epochs=5 dense_adj=True fast_neg=True)",
        "(ClientAppActor pid=1) [timing] client 1 round 1 fit: setup 0.40s train 4.00s (epochs=5 dense_adj=True fast_neg=True)",
        "[2026-09-21 06:01:00,000][flwr][INFO] - aggregate_fit: received 2 results and 0 failures",
        "[2026-09-21 06:01:01,000][flwr][INFO] - fit progress: (1, 0.0, {}, 61.0)",
        "[2026-09-21 06:01:01,500][flwr][INFO] - configure_evaluate: strategy sampled 2 clients (out of 2)",
        "[2026-09-21 06:01:11,500][flwr][INFO] - aggregate_evaluate: received 2 results and 0 failures",
        "[2026-09-21 06:01:11,600][flwr][INFO] - [ROUND 2]",
    ]))
    bd = jp.phase_breakdown(str(log))
    assert len(bd) == 1
    r = bd.iloc[0]
    assert r["fit_s"] == 60 and r["agg_eval_s"] == 1 and r["client_eval_s"] == 10
    assert r["train_s"] == 10 and r["overhead_s"] == 50


def test_caga_routing_maps_flower_node_ids_to_partition_ids(tmp_path):
    """Old runs logged random 64-bit Flower node ids as CAGA keys; routing must still land on
    partition ids.
    """
    rng = np.random.default_rng(1)
    node_ids = [str(int(x)) for x in rng.integers(10**18, 2 * 10**18, size=10)]
    for new_style in (False, True):
        root = tmp_path / ("new" if new_style else "old"); d = root / "joint_caga_seed0"; os.makedirs(d)
        keys = [str(i) for i in range(10)] if new_style else node_ids
        rows = []
        for r in range(1, 101):
            verd = ["rejected", "rejected", "phi_gated", "phi_gated"] + ["accepted"] * 6
            rows.append(dict(round=r, caga_client_keys=json.dumps(keys), caga_verdict=str(verd),
                             caga_scale_phi=json.dumps([1.0] * 10)))
        pd.DataFrame(rows).to_csv(d / "results_x_seed0.csv", index=False)
        t = jp.caga_routing(str(root))
        assert t.loc["poisoned", "rejected"] == 1.0 and t.loc["drifted", "phi_gated"] == 1.0
        assert t.loc["honest", "flagged"] == 0.0


def test_fltrust_tier_is_optional_and_uses_root_device():
    assert all(r["method"] != "fltrust" for r in jp.make_runs())
    runs = jp.make_runs(tiers=(4,))
    assert len(runs) == 5 * 4 * 2 and {r["method"] for r in runs} == {"fltrust", "rootclient"}
    s = " ".join(jp.overrides_for(runs[0], "/tmp/x"))
    assert "defense.mode=fltrust" in s and "defense.root_device=cuda" in s


def test_drift_defaults_to_severe_and_ablation_disables_block_gating():
    _, d = jp.build_specs()
    assert all(e["feature_dims"] == list(range(50)) and e["target_node_frac"] == 0.75 for e in d["d10"])
    _, base = jp.build_specs(dict(jp.PROTOCOL, drift_strength="base"))
    assert base["d10"][0]["feature_dims"] == [0, 1, 2, 3]
    r = jp.make_runs(tiers=(5,))[0]
    assert "defense.caga_gate_blocks=False" in jp.overrides_for(r, "/tmp/x")
