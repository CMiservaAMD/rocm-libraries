# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Load the maintained model (model.yaml, rules.yaml, injections.yaml) and extra rule files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODEL_FILES = ("model.yaml", "rules.yaml", "injections.yaml")


class ModelError(Exception):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    form: str
    # Each conjunction maps an axis to the forbidden values of that axis.
    conjs: tuple
    source_file: str


@dataclass
class Model:
    axes: list
    values: dict
    parent: dict
    children: dict
    base_cases: dict
    rules: list
    non_exclusion_rules: list
    hints: dict
    input_files: dict = field(default_factory=dict)
    injection_counts: dict = field(default_factory=dict)

    def is_active(self, row: dict, axis: str) -> bool:
        if axis not in self.parent:
            return True
        pax, pvals = self.parent[axis]
        return self.is_active(row, pax) and row.get(pax) in pvals

    def violated_rules(self, row: dict) -> list:
        """Return the ids of the exclusion rules that the full row breaks."""
        out = []
        for r in self.rules:
            for c in r.conjs:
                if all(
                    row.get(a) is not None and row.get(a) in vs for a, vs in c.items()
                ):
                    out.append(r.id)
                    break
        return out

    def structure_errors(self, row: dict) -> list:
        errs = []
        for a in self.axes:
            v = row.get(a)
            active = self.is_active(row, a)
            if active and v is None:
                errs.append(f"{a}: active but empty")
            elif not active and v is not None:
                errs.append(f"{a}: inactive but set to {v}")
            elif v is not None and v not in self.values[a]:
                errs.append(f"{a}: unknown value {v}")
        return errs


def _check_axis_values(axes_values: dict, where: str, mapping: dict) -> None:
    for ax, vals in mapping.items():
        if ax not in axes_values:
            raise ModelError(f"{where}: unknown axis {ax}")
        for v in vals if isinstance(vals, (list, tuple, set, frozenset)) else [vals]:
            if v not in axes_values[ax]:
                raise ModelError(f"{where}: unknown value {ax}={v}")


def _parse_rules(doc: dict, source: str, values: dict):
    exclusions, others = [], []
    seen = set()
    for r in doc.get("rules") or []:
        rid = r["id"]
        if rid in seen:
            raise ModelError(f"{source}: duplicate rule id {rid}")
        seen.add(rid)
        for key in ("kind", "reason"):
            if key not in r:
                raise ModelError(f"{source}: rule {rid} has no {key}")
        raw = []
        if "forbid" in r:
            raw.append(r["forbid"])
        raw.extend(r.get("forbid_any") or [])
        if not raw:
            others.append({"id": rid, "kind": r["kind"], "form": r.get("form", "")})
            continue
        conjs = []
        for c in raw:
            _check_axis_values(values, f"{source}: rule {rid}", c)
            conjs.append({ax: frozenset(vs) for ax, vs in c.items()})
        exclusions.append(
            Rule(rid, r["kind"], r.get("form", "exclusion"), tuple(conjs), source)
        )
    return exclusions, others


def load_model(model_dir: Path, extra_rule_files=()) -> Model:
    model_dir = Path(model_dir)
    m = load_yaml(model_dir / "model.yaml")
    axes, values, parent, children = [], {}, {}, {}
    for a in m["axes"]:
        aid = a["id"]
        if aid in values:
            raise ModelError(f"duplicate axis {aid}")
        ids = [v["id"] for v in a["values"]]
        if len(set(ids)) != len(ids):
            raise ModelError(f"duplicate value in axis {aid}")
        axes.append(aid)
        values[aid] = ids
        children[aid] = []
    for a in m["axes"]:
        if a["kind"] == "child":
            ((pax, pvals),) = a["parent"].items()
            _check_axis_values(values, f"axis {a['id']} parent", {pax: pvals})
            parent[a["id"]] = (pax, frozenset(pvals))
            children[pax].append(a["id"])

    bases = {}
    for name, b in (m.get("base_cases") or {}).items():
        vals = b["values"]
        _check_axis_values(
            values, f"base case {name}", {k: [v] for k, v in vals.items()}
        )
        bases[name] = dict(vals)

    rules, others = [], []
    rule_docs = [("rules.yaml", model_dir / "rules.yaml")]
    rule_docs += [(str(Path(p).resolve()), Path(p)) for p in extra_rule_files]
    ids = set()
    for label, path in rule_docs:
        ex, ot = _parse_rules(load_yaml(path), label, values)
        for r in ex:
            if r.id in ids:
                raise ModelError(f"rule id {r.id} appears in two rule files")
            ids.add(r.id)
        rules.extend(ex)
        others.extend(ot)

    inj = load_yaml(model_dir / "injections.yaml")
    inj_counts = _check_injections(inj, values, bases)

    model = Model(
        axes=axes,
        values=values,
        parent=parent,
        children=children,
        base_cases={},
        rules=rules,
        non_exclusion_rules=others,
        hints=m.get("generator_hints") or {},
        injection_counts=inj_counts,
    )
    for name, vals in bases.items():
        row = {a: vals.get(a) for a in axes}
        errs = model.structure_errors(row)
        if errs:
            raise ModelError(f"base case {name}: {errs}")
        broken = model.violated_rules(row)
        if broken:
            raise ModelError(f"base case {name} breaks rules {broken}")
        model.base_cases[name] = row
    model.input_files = {f"model/{n}": sha256_file(model_dir / n) for n in MODEL_FILES}
    for label, path in rule_docs[1:]:
        model.input_files[label] = sha256_file(path)
    return model


def _check_injections(doc: dict, values: dict, bases: dict) -> dict:
    counts = {}
    for section in (
        "rejection_probes",
        "path_variants",
        "frontend_rejection_controls",
        "conditional_probes",
    ):
        n = 0
        for entry in doc.get(section) or []:
            items = list(entry.get("probes") or entry.get("variants") or [entry])
            if "equivalent_cpp_request" in entry:
                items.append(entry["equivalent_cpp_request"])
            for p in items:
                if not isinstance(p, dict):
                    continue
                start = p.get("start_from", entry.get("start_from"))
                if start is not None and start not in bases:
                    raise ModelError(f"injections {section}: unknown base {start}")
                changes = p.get("changes") or {}
                _check_axis_values(
                    values,
                    f"injections {section} {p.get('id', entry.get('id'))}",
                    {k: [v] for k, v in changes.items()},
                )
                n += 1
        counts[section] = n
    return counts
