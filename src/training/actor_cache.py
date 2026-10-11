"""Per-process cache of the client trainers (opt-in: perf.actor_cache=true)"""
from __future__ import annotations

import pickle
from typing import Any, Dict, List

_TRAINERS: Dict[str, List[Any]] = {}


def get_trainers(path: str) -> List[Any]:
    """Load the pickled trainer list once per process and return the same list on every later call."""
    trainers = _TRAINERS.get(path)
    if trainers is None:
        with open(path, "rb") as f:
            trainers = pickle.load(f)
        _TRAINERS[path] = trainers
    return trainers
