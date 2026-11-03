# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Command line: generate the case sets and their manifest from a maintained model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import platform
import random
import sys
from pathlib import Path

import yaml

from . import class_values, modes
from .model import Model, ModelError, _check_axis_values, load_model, sha256_file
from .pairs import candidate_pairs, feasible_pairs
from .solver import Solver

TOOL_NAME = "sdpa_case_flow_poc"
TOOL_VERSION = "0.1.0"

# Files that make up the generator. The content hash covers these files only.
GENERATOR_FILES = (
    "__init__.py",
    "__main__.py",
    "model.py",
    "class_values.py",
    "solver.py",
    "pairs.py",
    "modes.py",
    "generate.py",
)

ALL_MODES = (
    "one-at-a-time",
    "pairwise",
    "pairwise-per-value",
    "every-pair-isolated",
    "random-sample",
)


def _requirements(model: Model, axis: str) -> dict:
    """Map each ancestor of `axis` to the values it must hold for `axis` to be active."""
    req = {}
    a = axis
    while a in model.parent:
        pax, pvals = model.parent[a]
        req[pax] = set(pvals) if pax not in req else req[pax] & set(pvals)
        a = pax
    return req


def structural_pair_count(model: Model, skip_axes=()) -> int:
    """Count structurally possible pairs per axis pair, from the parent/child links alone."""
    axes = [a for a in model.axes if a not in skip_axes]
    reqs = {a: _requirements(model, a) for a in axes}
    total = 0
    for a, b in itertools.combinations(axes, 2):
        for x in model.values[a]:
            for y in model.values[b]:
                need = {}
                ok = True
                for ax, allowed in itertools.chain(
                    reqs[a].items(), reqs[b].items(), ((a, {x}), (b, {y}))
                ):
                    need[ax] = (
                        set(allowed) if ax not in need else need[ax] & set(allowed)
                    )
                    if not need[ax]:
                        ok = False
                        break
                total += ok
    return total


def analyse(model: Model, solver: Solver) -> dict:
    """Return valid pairs, excluded pairs with the rule ids that exclude them, and counts."""
    cands = candidate_pairs(model)
    structural = feasible_pairs(Solver(model, rules=[]), cands)
    arithmetic = structural_pair_count(model)
    if arithmetic != len(structural):
        raise AssertionError(
            f"structural pairs: arithmetic {arithmetic} != solver {len(structural)}"
        )
    valid = feasible_pairs(solver, structural)
    valid_set = set(valid)
    excluded = [p for p in structural if p not in valid_set]

    single_rule = {r.id: Solver(model, rules=[r]) for r in model.rules}
    all_but = {}

    def without(rid):
        if rid not in all_but:
            all_but[rid] = Solver(model, rules=[r for r in model.rules if r.id != rid])
        return all_but[rid]

    def attribute(fixed: dict):
        sole = sorted(rid for rid, s in single_rule.items() if not s.feasible(fixed))
        if sole:
            return sole, (sole[0] if len(sole) == 1 else "any-of:" + "+".join(sole))
        needed = sorted(r.id for r in model.rules if without(r.id).feasible(fixed))
        return [], "jointly:" + ("+".join(needed) if needed else "unresolved")

    excluded_values = []
    for a in model.axes:
        for v in model.values[a]:
            if not solver.feasible({a: v}):
                rules, label = attribute({a: v})
                excluded_values.append({"axis": a, "value": v, "excluded_by": label})

    by_rule = {r.id: 0 for r in model.rules}
    by_label = {}
    excluded_list = []
    for f, g in excluded:
        rules, label = attribute({f[0]: f[1], g[0]: g[1]})
        for rid in rules:
            by_rule[rid] += 1
        by_label[label] = by_label.get(label, 0) + 1
        excluded_list.append((f, g, label))

    return {
        "structural_pairs_arithmetic": arithmetic,
        "structural_pairs_solver": len(structural),
        "valid": valid,
        "excluded": excluded_list,
        "excluded_values": excluded_values,
        "excluded_by_rule_alone": by_rule,
        "excluded_by_label": dict(sorted(by_label.items())),
    }


def every_pair_isolated(
    model: Model, solver: Solver, valid_pairs, base_name: str, oat_rows
) -> tuple:
    """One row per valid pair, base values elsewhere.

    A pair with a base value gets no row of its own when the one-at-a-time row of its other value,
    or the base row, already holds it.
    """
    base = model.base_cases[base_name]
    held = {}
    for meta, row in oat_rows:
        if meta["base"] != base_name:
            continue
        if meta["varied"] == "-":
            held[None] = row
        else:
            a, v = meta["varied"].split("=", 1)
            held[(a, v)] = row
    rows = []
    with_base_value = 0
    for f, g in valid_pairs:
        f_base, g_base = base[f[0]] == f[1], base[g[0]] == g[1]
        if f_base or g_base:
            key = None if f_base and g_base else (g if f_base else f)
            r = held.get(key)
            if r is not None and r[f[0]] == f[1] and r[g[0]] == g[1]:
                with_base_value += 1
                continue
        row = solver.complete({f[0]: f[1], g[0]: g[1]}, prefs=base)
        if row is None:
            raise AssertionError(f"valid pair {f} {g} has no row")
        meta = {
            "base": base_name,
            "pair": f"{f[0]}={f[1]}|{g[0]}={g[1]}",
            "also_changed": modes._diff(model, base, row, skip=(f[0], g[0])),
        }
        rows.append((meta, row))
    return rows, with_base_value


def random_sample(model: Model, solver: Solver, count: int, rng: random.Random) -> list:
    rows = []
    for _ in range(count):
        row = solver.complete({}, prefs={}, rng=rng)
        rows.append(({}, row))
    return rows


def load_bases_file(model: Model, path: Path) -> None:
    """Add named base cases from a file with the model.yaml `base_cases` layout."""
    doc = load_yaml(path)
    for name, b in (doc.get("base_cases") or {}).items():
        if name in model.base_cases:
            raise ModelError(f"base case {name} defined twice")
        vals = b["values"]
        _check_axis_values(
            model.values, f"base case {name}", {k: [v] for k, v in vals.items()}
        )
        row = {a: vals.get(a) for a in model.axes}
        errs = model.structure_errors(row)
        if errs:
            raise ModelError(f"base case {name}: {errs}")
        broken = model.violated_rules(row)
        if broken:
            raise ModelError(f"base case {name} breaks rules {broken}")
        model.base_cases[name] = row
    model.input_files[str(Path(path).resolve())] = sha256_file(path)


def content_hash() -> tuple:
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    per_file = {}
    for name in GENERATOR_FILES:
        data = (here / name).read_bytes()
        per_file[name] = hashlib.sha256(data).hexdigest()
        h.update(name.encode() + b"\0" + data + b"\0")
    return h.hexdigest(), per_file


def render_csv(model: Model, rows, meta_cols, prefix: str) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["case_id", *meta_cols, *model.axes])
    width = max(4, len(str(len(rows))))
    for i, (meta, row) in enumerate(rows, start=1):
        errs = model.structure_errors(row)
        broken = model.violated_rules(row)
        if errs or broken:
            raise AssertionError(f"{prefix}-{i}: structure {errs} rules {broken}")
        w.writerow(
            [
                f"{prefix}-{i:0{width}d}",
                *(meta.get(c, "") for c in meta_cols),
                *("-" if row[a] is None else row[a] for a in model.axes),
            ]
        )
    return buf.getvalue().encode("utf-8")


def write(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def lower_bound(model: Model, pairs) -> dict:
    """The largest number of valid pairs between two axes: no covering set can be smaller."""
    per_axis_pair = {}
    for f, g in pairs:
        k = (f[0], g[0])
        per_axis_pair[k] = per_axis_pair.get(k, 0) + 1
    (a, b), n = max(per_axis_pair.items(), key=lambda kv: (kv[1], kv[0]))
    return {"cases": n, "axes": [a, b]}


def stratum_pairs(model: Model, solver: Solver, fixed: dict) -> list:
    cands = candidate_pairs(model, skip_axes=tuple(fixed))
    return feasible_pairs(solver, cands, fixed=fixed)


def parse_args(argv):
    ap = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="kept outputs: manifest, one-at-a-time, pairwise sets",
    )
    ap.add_argument(
        "--large-out-dir",
        type=Path,
        help="large outputs: every-pair-isolated, random sample (default: --out-dir)",
    )
    ap.add_argument("--seed", type=int, default=2893)
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument(
        "--rules-extra",
        action="append",
        default=[],
        type=Path,
        help="extra rule file in the rules.yaml format (for example engine:<name> rules)",
    )
    ap.add_argument(
        "--bases-file",
        type=Path,
        help="extra named base cases (model.yaml base_cases layout)",
    )
    ap.add_argument(
        "--bases", default="shared", help="comma-separated base names for one-at-a-time"
    )
    ap.add_argument("--isolated-base", default="shared")
    ap.add_argument("--random-count", type=int, default=1000)
    ap.add_argument(
        "--candidates", type=int, default=2, help="candidate rows per pairwise step"
    )
    ap.add_argument(
        "--excluded-pairs-out",
        type=Path,
        help="also write every excluded pair with its rule id",
    )
    ap.add_argument(
        "--skip-second-strength",
        action="store_true",
        help="skip the second extra-strength count",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    selected = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in selected:
        if m not in ALL_MODES:
            print(f"unknown mode {m}", file=sys.stderr)
            return 2

    model = load_model(args.model_dir, args.rules_extra)
    if args.bases_file:
        load_bases_file(model, args.bases_file)
    solver = Solver(model)
    rng = random.Random(args.seed)
    digest, per_file = content_hash()

    analysis = analyse(model, solver)
    valid = analysis["valid"]

    manifest = {
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
            "content_sha256": digest,
            "files": per_file,
            "python": platform.python_version(),
            "pyyaml": yaml.__version__,
        },
        "seed": args.seed,
        "command": {
            "modes": selected,
            "bases": args.bases,
            "isolated_base": args.isolated_base,
            "random_count": args.random_count,
            "candidates": args.candidates,
        },
        "inputs": model.input_files,
        "model": {
            "axes": len(model.axes),
            "child_axes": len(model.parent),
            "values": sum(len(v) for v in model.values.values()),
            "independent_components": len(solver.components),
            "injections_not_generated": model.injection_counts,
        },
        "rules": {
            "exclusion_rules": len(model.rules),
            "other_rules_removing_nothing": len(model.non_exclusion_rules),
            "exclusion_by_kind": _count_by(r.kind for r in model.rules),
            "rules_with_a_conjunction_over_3_or_more_axes": sorted(
                r.id for r in model.rules if max(len(c) for c in r.conjs) >= 3
            ),
        },
        "pairs": {
            "structural_pairs_arithmetic": analysis["structural_pairs_arithmetic"],
            "valid_pairs": len(valid),
            "excluded_pairs": len(analysis["excluded"]),
            "check_valid_plus_excluded_equals_structural": len(valid)
            + len(analysis["excluded"])
            == analysis["structural_pairs_arithmetic"],
            "excluded_values": analysis["excluded_values"],
            "excluded_pairs_by_rule_alone": analysis["excluded_by_rule_alone"],
            "excluded_pairs_partition": analysis["excluded_by_label"],
            "definitions": {
                "structural": "two values of different axes that parent/child links allow in one row",
                "valid": "a structural pair that at least one rule-abiding row contains",
                "by_rule_alone": "excluded pairs that this rule excludes with every other rule removed; a pair can count under several rules",
                "partition": "each excluded pair once: one rule id; any-of:<ids> when each listed rule alone excludes it; jointly:<ids> when only the listed rules together exclude it",
            },
        },
        "outputs": {},
    }

    if args.excluded_pairs_out:
        lines = ["axis_a,value_a,axis_b,value_b,excluded_by"]
        lines += [
            f"{f[0]},{f[1]},{g[0]},{g[1]},{lab}" for f, g, lab in analysis["excluded"]
        ]
        info = write(args.excluded_pairs_out, ("\n".join(lines) + "\n").encode())
        manifest["outputs"]["excluded-pairs"] = {
            "path": str(args.excluded_pairs_out.resolve()),
            "rows": len(analysis["excluded"]),
            **info,
        }

    out = manifest["outputs"]
    info = write(
        args.out_dir / class_values.FILE_NAME,
        class_values.render(class_values.build(model, args.model_dir)),
    )
    out["class-to-value"] = {
        "file": class_values.FILE_NAME,
        "schema_version": class_values.SCHEMA_VERSION,
        **info,
    }

    if "one-at-a-time" in selected:
        names = [n.strip() for n in args.bases.split(",") if n.strip()]
        rows, excl = modes.one_at_a_time(model, solver, names)
        info = write(
            args.out_dir / "one-at-a-time.csv",
            render_csv(model, rows, ["base", "varied", "also_changed"], "oat"),
        )
        out["one-at-a-time"] = {
            "file": "one-at-a-time.csv",
            "rows": len(rows),
            **info,
            "bases": names,
            "rows_with_extra_changes": sum(1 for m, _ in rows if m["also_changed"]),
            "values_no_row_can_hold": [f"{b}:{a}={v}" for b, a, v in excl],
        }

    if "pairwise" in selected:
        rows = modes.pairwise(model, solver, valid, rng, candidates=args.candidates)
        info = write(args.out_dir / "pairwise.csv", render_csv(model, rows, [], "pw"))
        out["pairwise"] = {
            "file": "pairwise.csv",
            "rows": len(rows),
            **info,
            "valid_pairs": len(valid),
            "lower_bound": lower_bound(model, valid),
        }

    per_value_axis = model.hints.get("per_value_pairwise_axis", "dtype_q")
    stratum_cache = {}
    if "pairwise-per-value" in selected:
        sets = {}
        total = 0
        for v in model.values[per_value_axis]:
            fixed = {per_value_axis: v}
            if not solver.feasible(fixed):
                sets[v] = {"rows": 0, "note": "no valid row has this value"}
                continue
            sp = stratum_pairs(model, solver, fixed)
            stratum_cache[v] = sp
            rows = modes.pairwise(
                model, solver, sp, rng, fixed=fixed, candidates=args.candidates
            )
            name = f"pairwise-{v}.csv"
            info = write(args.out_dir / name, render_csv(model, rows, [], f"pw-{v}"))
            sets[v] = {
                "file": name,
                "rows": len(rows),
                **info,
                "valid_pairs": len(sp),
                "lower_bound": lower_bound(model, sp),
            }
            total += len(rows)
        out["pairwise-per-value"] = {
            "axis": per_value_axis,
            "total_rows": total,
            "sets": sets,
        }

    if "every-pair-isolated" in selected:
        oat_rows, _ = modes.one_at_a_time(model, solver, [args.isolated_base])
        rows, skipped = every_pair_isolated(
            model, solver, valid, args.isolated_base, oat_rows
        )
        path = (args.large_out_dir or args.out_dir) / "every-pair-isolated.csv"
        info = write(
            path, render_csv(model, rows, ["base", "pair", "also_changed"], "iso")
        )
        out["every-pair-isolated"] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            **info,
            "base": args.isolated_base,
            "valid_pairs_held_by_one_at_a_time_rows": skipped,
            "note": "a pair with a base value has its own row only when the one-at-a-time row of its other value (or the base row) does not hold it",
        }

    if "random-sample" in selected:
        rows = random_sample(model, solver, args.random_count, rng)
        path = (args.large_out_dir or args.out_dir) / "random-sample.csv"
        info = write(path, render_csv(model, rows, [], "rnd"))
        out["random-sample"] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            **info,
            "method": "per row: shuffle every axis's value order, then take the first valid row the solver finds; not uniform over valid rows",
        }

    if not args.skip_second_strength:
        second = model.hints.get("second_strength_candidate", "dim_order_q")
        per = {}
        total = 0
        for v in model.values[per_value_axis]:
            for w in model.values[second]:
                fixed = {per_value_axis: v, second: w}
                if not solver.feasible(fixed):
                    continue
                sp = stratum_pairs(model, solver, fixed)
                srng = random.Random(f"{args.seed}:{v}:{w}")
                n = len(
                    modes.pairwise(
                        model, solver, sp, srng, fixed=fixed, candidates=args.candidates
                    )
                )
                per[f"{v}+{w}"] = {"rows": n, "valid_pairs": len(sp)}
                total += n
        manifest["second_extra_strength_count_only"] = {
            "axes": [per_value_axis, second],
            "strata": len(per),
            "total_rows": total,
            "per_stratum": per,
            "note": "one complete pairwise set per value combination of the two axes; counted, not written",
        }

    manifest_bytes = yaml.safe_dump(
        manifest, sort_keys=False, width=110, allow_unicode=True
    ).encode("utf-8")
    write(args.out_dir / "manifest.yaml", manifest_bytes)
    _summary(manifest)
    return 0


def _count_by(items) -> dict:
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def _summary(manifest: dict) -> None:
    p = manifest["pairs"]
    print(
        f"valid pairs {p['valid_pairs']}, excluded {p['excluded_pairs']}, "
        f"structural {p['structural_pairs_arithmetic']}"
    )
    for k, v in manifest["outputs"].items():
        if "rows" in v:
            print(f"{k}: {v['rows']} rows")
        elif "total_rows" in v:
            print(f"{k}: {v['total_rows']} rows")
    s = manifest.get("second_extra_strength_count_only")
    if s:
        print(
            f"second extra-strength axis: {s['total_rows']} rows over {s['strata']} strata"
        )


if __name__ == "__main__":
    sys.exit(main())
