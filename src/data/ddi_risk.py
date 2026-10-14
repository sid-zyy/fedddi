"""Clinical risk classes for DrugBank interactions, from their description templates.

A template is the interaction description with both drugs' names replaced by ``#DRUG``. Ordered rules
in ``conf/ddi_risk_map.yaml`` map each template to an interaction type and a risk class; the first rule
that matches wins and anything unmatched is ``low``.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

RISK_NAMES = ("low", "moderate", "high")
RISK_CODE = {name: i for i, name in enumerate(RISK_NAMES)}
DEFAULT_MAP = os.path.join(os.path.dirname(__file__), "..", "..", "conf", "ddi_risk_map.yaml")
FALLBACK_TYPE = "other"


def make_template(description: str, names: Iterable[str]) -> str:
    """Replace each drug name in the description with ``#DRUG``, longest name first."""
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        description = description.replace(name, "#DRUG")
    return description


class RiskMap:
    """Ordered ``{template | pattern, type, risk}`` rules; ``template`` is an exact match,
    ``pattern`` a case-insensitive regular-expression search."""

    def __init__(self, rules: Sequence[dict], path: str = "") -> None:
        self.path = path
        self.rules: List[Tuple[str, object, str, int]] = []
        for i, r in enumerate(rules):
            if r.get("risk") not in RISK_CODE:
                raise ValueError(f"rule {i}: risk must be one of {RISK_NAMES}, got {r.get('risk')!r}")
            if ("template" in r) == ("pattern" in r):
                raise ValueError(f"rule {i}: give exactly one of 'template' or 'pattern'")
            key = r["template"] if "template" in r else re.compile(r["pattern"], re.IGNORECASE)
            self.rules.append(("template" if "template" in r else "pattern", key, str(r["type"]),
                               RISK_CODE[r["risk"]]))
        self.types: List[str] = sorted({t for _, _, t, _ in self.rules} | {FALLBACK_TYPE})
        self._cache: Dict[str, Tuple[str, int]] = {}

    @classmethod
    def load(cls, path: str = DEFAULT_MAP) -> "RiskMap":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f)["rules"], path)

    def classify(self, template: str) -> Tuple[str, int]:
        """(type, risk code) for one template."""
        hit = self._cache.get(template)
        if hit is None:
            hit = (FALLBACK_TYPE, RISK_CODE["low"])
            for kind, key, typ, risk in self.rules:
                if (template == key) if kind == "template" else key.search(template):  # type: ignore[union-attr]
                    hit = (typ, risk)
                    break
            self._cache[template] = hit
        return hit

    def type_id(self, typ: str) -> int:
        return self.types.index(typ)
