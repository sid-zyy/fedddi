"""Torch-free test of src/training/actor_cache.py (loaded by path: the package __init__ imports
torch).
"""
import importlib.util, os, pickle, tempfile

_spec = importlib.util.spec_from_file_location(
    "actor_cache", os.path.join(os.path.dirname(__file__), "..", "src", "training", "actor_cache.py"))
ac = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ac)


def test_trainers_are_loaded_once_per_process_and_reused():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.pkl")
        with open(path, "wb") as f:
            pickle.dump([{"cid": i, "edges": list(range(5))} for i in range(3)], f)
        a = ac.get_trainers(path)
        os.remove(path)
        b = ac.get_trainers(path)
        assert a is b and [t["cid"] for t in b] == [0, 1, 2]
        a[1]["edges"] = ["mutated"]
        assert ac.get_trainers(path)[1]["edges"] == ["mutated"]
