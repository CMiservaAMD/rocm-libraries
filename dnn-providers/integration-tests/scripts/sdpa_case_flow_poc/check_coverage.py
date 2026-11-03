#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage checker for generated case sets. It recomputes the valid pairs from the model files.

Subcommands
  pairwise    every valid pair appears in at least one row (with --fix: under the fixed values).
  isolated    every valid pair has its own row: an every-pair-isolated row when both values
              differ from the base. A pair with a base value is held by the one-at-a-time row of
              its other value (or the base row), or else by an every-pair-isolated row.
  oat         every value that a valid row can hold is varied from the base in its own row, and
              each row differs from the base only in the axes it names.
  accounting  structural pairs = valid pairs + excluded pairs, and the excluded-pair list and its
              rule ids match the model.

A pair is valid when some rule-abiding row contains it. Pairs found in a checked row are valid.
Every other pair is decided by a search over the model, so a valid pair missing from the set is
found and named. Exit status 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_common import Spec, read_cases, row_pairs  # noqa: E402

MAX_LISTED = 20


def fmt_pair(p) -> str:
    (a, x), (b, y) = p
    return f"{a}={x} + {b}={y}"


def report(title: str, problems: list, facts: list) -> int:
    status = "PASS" if not problems else "FAIL"
    print(f"{status} {title}")
    for f in facts:
        print(f"  {f}")
    for p in problems[:MAX_LISTED]:
        print(f"  {p}")
    if len(problems) > MAX_LISTED:
        print(f"  ... {len(problems) - MAX_LISTED} more")
    return 1 if problems else 0


def load_rows(spec: Spec, path: Path, problems: list) -> list:
    """Read a CSV. Invalid rows are reported as problems and left out of the returned rows."""
    rows = list(read_cases(path, spec))
    if not rows:
        problems.append(f"{path.name}: no rows")
    good = []
    for case_id, meta, row in rows:
        probs = spec.row_problems(row)
        for p in probs:
            problems.append(f"{path.name} {case_id}: invalid row: {p}")
        if not probs:
            good.append((case_id, meta, row))
    return good


def parse_fix(items) -> dict:
    out = {}
    for it in items or []:
        a, v = it.split("=", 1)
        out[a] = v
    return out


def valid_pairs(spec: Spec, rows_list, fixed: dict) -> tuple:
    """Return (valid pairs, pairs covered by the rows). Every row passed in is valid."""
    covered = set()
    for rows in rows_list:
        for _, _, row in rows:
            if all(row.get(a) == v for a, v in fixed.items()):
                for p in row_pairs(spec, row):
                    if p[0][0] not in fixed and p[1][0] not in fixed:
                        covered.add(p)
    valid = set(covered)
    for p in spec.candidate_pairs(skip=tuple(fixed)):
        if p in covered:
            continue
        (a, x), (b, y) = p
        if spec.satisfiable({**fixed, a: x, b: y}):
            valid.add(p)
    return valid, covered


def cmd_pairwise(spec: Spec, args) -> int:
    problems = []
    fixed = parse_fix(args.fix)
    rows = load_rows(spec, args.csv, problems)
    for case_id, _, row in rows:
        for a, v in fixed.items():
            if row.get(a) != v:
                problems.append(
                    f"{case_id}: {a}={row.get(a)}, expected the fixed value {v}"
                )
    valid, covered = valid_pairs(spec, [rows], fixed)
    missing = sorted(valid - covered)
    problems += [f"valid pair not covered: {fmt_pair(p)}" for p in missing]
    facts = [
        f"rows {len(rows)}",
        f"valid pairs (recomputed) {len(valid)}",
        f"covered {len(valid & covered)}",
        f"missing {len(missing)}",
    ]
    if args.expect_valid is not None and args.expect_valid != len(valid):
        problems.append(f"valid pairs {len(valid)} != manifest {args.expect_valid}")
    suffix = f" with {fixed}" if fixed else ""
    return report(f"pairwise coverage {args.csv.name}{suffix}", problems, facts)


def differs_only_in(spec: Spec, base: dict, row: dict, allowed: set) -> list:
    return [a for a in spec.axes if row[a] != base[a] and a not in allowed]


def parse_changes(text: str) -> dict:
    out = {}
    for item in filter(None, (text or "").split(";")):
        a, v = item.split("=", 1)
        out[a] = None if v == "-" else v
    return out


def cmd_oat(spec: Spec, args) -> int:
    problems = []
    base = spec.base_row(args.base)
    rows = [
        r for r in load_rows(spec, args.csv, problems) if r[1].get("base") == args.base
    ]
    varied = {}
    base_rows = 0
    for case_id, meta, row in rows:
        if meta["varied"] == "-":
            base_rows += 1
            if row != base:
                problems.append(
                    f"{case_id}: marked as the base row but differs from base {args.base}"
                )
            continue
        a, v = meta["varied"].split("=", 1)
        if row[a] != v:
            problems.append(f"{case_id}: varied {a}={v} but the row has {row[a]}")
        changes = parse_changes(meta["also_changed"])
        for ca, cv in changes.items():
            if row[ca] != cv:
                problems.append(
                    f"{case_id}: also_changed {ca}={cv} but the row has {row[ca]}"
                )
        extra = differs_only_in(spec, base, row, {a, *changes})
        if extra:
            problems.append(
                f"{case_id}: differs from the base in unlisted axes {extra}"
            )
        varied.setdefault((a, v), case_id)
    if base_rows != 1:
        problems.append(f"{base_rows} base rows, expected 1")
    expected = 0
    unreachable = []
    for a in spec.axes:
        for v in spec.values[a]:
            if v == base[a]:
                continue
            if spec.satisfiable({a: v}):
                expected += 1
                if (a, v) not in varied:
                    problems.append(f"value with no one-at-a-time row: {a}={v}")
            else:
                unreachable.append(f"{a}={v}")
                if (a, v) in varied:
                    problems.append(f"row varies {a}={v}, which no valid row can hold")
    facts = [
        f"rows {len(rows)}",
        f"values that need a row {expected}",
        f"values no valid row can hold {len(unreachable)}",
    ]
    return report(
        f"one-at-a-time coverage {args.csv.name} (base {args.base})", problems, facts
    )


def cmd_isolated(spec: Spec, args) -> int:
    problems = []
    base = spec.base_row(args.base)
    iso = load_rows(spec, args.csv, problems)
    oat = [
        r for r in load_rows(spec, args.oat, problems) if r[1].get("base") == args.base
    ]
    own = {}
    for case_id, meta, row in iso:
        fa, fb = meta["pair"].split("|")
        a, x = fa.split("=", 1)
        b, y = fb.split("=", 1)
        if row[a] != x or row[b] != y:
            problems.append(
                f"{case_id}: names pair {meta['pair']} but does not contain it"
            )
        changes = parse_changes(meta["also_changed"])
        extra = differs_only_in(spec, base, row, {a, b, *changes})
        if extra:
            problems.append(
                f"{case_id}: differs from the base in unlisted axes {extra}"
            )
        key = ((a, x), (b, y))
        if key in own:
            problems.append(
                f"{case_id}: pair {meta['pair']} already has row {own[key]}"
            )
        own[key] = case_id
    oat_rows = {}
    base_row = None
    for case_id, meta, row in oat:
        if meta["varied"] == "-":
            base_row = row
        else:
            a, v = meta["varied"].split("=", 1)
            oat_rows[(a, v)] = row
    valid, _ = valid_pairs(spec, [iso, oat], {})
    n_own = n_oat = n_base = n_own_base = 0
    for p in sorted(valid):
        f, g = p
        f_base, g_base = base[f[0]] == f[1], base[g[0]] == g[1]
        if not f_base and not g_base:
            if p not in own:
                problems.append(f"valid pair with no isolated row: {fmt_pair(p)}")
            n_own += 1
        elif f_base and g_base:
            if base_row is None or base_row[f[0]] != f[1] or base_row[g[0]] != g[1]:
                if p not in own:
                    problems.append(
                        f"base pair in neither the base row nor its own row: {fmt_pair(p)}"
                    )
                n_own_base += 1
            else:
                n_base += 1
        else:
            other = g if f_base else f
            fixed_one = f if f_base else g
            r = oat_rows.get(other)
            if r is None or r[fixed_one[0]] != fixed_one[1]:
                if p not in own:
                    problems.append(
                        f"pair with one base value in neither its one-at-a-time row nor its own row: {fmt_pair(p)}"
                    )
                n_own_base += 1
            else:
                n_oat += 1
    stray = sorted(set(own) - valid)
    problems += [
        f"isolated row for a pair that is not valid: {fmt_pair(p)}" for p in stray
    ]
    facts = [
        f"isolated rows {len(iso)}",
        f"valid pairs (recomputed) {len(valid)}",
        f"needing their own isolated row {n_own}",
        f"held by one-at-a-time rows {n_oat}",
        f"with a base value but not held by a one-at-a-time row {n_own_base}",
        f"held by the base row {n_base}",
    ]
    return report(f"every-pair-isolated coverage {args.csv.name}", problems, facts)


def cmd_accounting(spec: Spec, args) -> int:
    problems = []
    with open(args.manifest, encoding="utf-8") as f:
        man = yaml.safe_load(f)
    witness_rows = []
    for path in args.rows:
        witness_rows.append(load_rows(spec, path, problems))
    seen = set()
    for rows in witness_rows:
        for _, _, row in rows:
            seen.update(row_pairs(spec, row))
    structural, valid = set(), set()
    no_rules = set()
    for p in spec.candidate_pairs():
        if p in seen:
            structural.add(p)
            valid.add(p)
            continue
        (a, x), (b, y) = p
        fixed = {a: x, b: y}
        if spec.satisfiable(fixed, rule_ids=no_rules):
            structural.add(p)
            if spec.satisfiable(fixed):
                valid.add(p)
    excluded = structural - valid

    listed = {}
    with open(args.excluded_csv, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            p = ((rec["axis_a"], rec["value_a"]), (rec["axis_b"], rec["value_b"]))
            if p in listed:
                problems.append(f"excluded list repeats {fmt_pair(p)}")
            listed[p] = rec["excluded_by"]
    for p in sorted(excluded - set(listed)):
        problems.append(f"excluded pair missing from the list: {fmt_pair(p)}")
    for p in sorted(set(listed) - excluded):
        problems.append(
            f"listed as excluded but {'valid' if p in valid else 'not structural'}: {fmt_pair(p)}"
        )

    label_counts = {}
    for p, label in sorted(listed.items()):
        label_counts[label] = label_counts.get(label, 0) + 1
        if p not in excluded:
            continue
        (a, x), (b, y) = p
        fixed = {a: x, b: y}
        if label.startswith("jointly:"):
            ids = set(label.split(":", 1)[1].split("+"))
            if spec.satisfiable(fixed, rule_ids=ids):
                problems.append(
                    f"{fmt_pair(p)}: rules {sorted(ids)} together do not exclude it"
                )
            for rid in ids:
                if not spec.satisfiable(fixed, rule_ids=set(spec.rule_ids) - {rid}):
                    problems.append(
                        f"{fmt_pair(p)}: excluded without {rid}, so {rid} is not needed"
                    )
        else:
            ids = (
                label.split(":", 1)[1].split("+")
                if label.startswith("any-of:")
                else [label]
            )
            for rid in ids:
                if rid not in spec.rule_ids:
                    problems.append(f"{fmt_pair(p)}: unknown rule id {rid}")
                elif spec.satisfiable(fixed, rule_ids={rid}):
                    problems.append(
                        f"{fmt_pair(p)}: rule {rid} alone does not exclude it"
                    )
            others = [r for r in spec.rule_ids if r not in ids]
            for rid in others:
                if not spec.satisfiable(fixed, rule_ids={rid}):
                    problems.append(
                        f"{fmt_pair(p)}: rule {rid} alone also excludes it but is not listed"
                    )

    mp = man["pairs"]
    arith = mp["structural_pairs_arithmetic"]
    if len(structural) != arith:
        problems.append(
            f"structural pairs: search {len(structural)} != manifest arithmetic {arith}"
        )
    if len(valid) + len(excluded) != arith:
        problems.append(
            f"valid {len(valid)} + excluded {len(excluded)} != structural {arith}"
        )
    if mp["valid_pairs"] != len(valid):
        problems.append(
            f"manifest valid_pairs {mp['valid_pairs']} != recomputed {len(valid)}"
        )
    if mp["excluded_pairs"] != len(excluded):
        problems.append(
            f"manifest excluded_pairs {mp['excluded_pairs']} != recomputed {len(excluded)}"
        )
    if mp["excluded_pairs_partition"] != label_counts:
        problems.append(
            "manifest excluded_pairs_partition differs from the counts of the excluded list"
        )
    if sum(label_counts.values()) != len(excluded):
        problems.append(
            f"partition sums to {sum(label_counts.values())}, excluded pairs {len(excluded)}"
        )
    facts = [
        f"structural pairs: search {len(structural)}, manifest arithmetic {arith}",
        f"valid {len(valid)} + excluded {len(excluded)} = {len(valid) + len(excluded)}",
        f"excluded list entries {len(listed)}, partition labels {len(label_counts)}",
    ]
    return report("excluded-pair accounting", problems, facts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--rules-extra", action="append", default=[], type=Path)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairwise")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--fix", action="append", help="axis=value held by every row")
    p.add_argument(
        "--expect-valid", type=int, help="valid-pair count the manifest reports"
    )
    p = sub.add_parser("oat")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--base", default="shared")
    p = sub.add_parser("isolated")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--oat", required=True, type=Path)
    p.add_argument("--base", default="shared")
    p = sub.add_parser("accounting")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--excluded-csv", required=True, type=Path)
    p.add_argument(
        "--rows",
        action="append",
        default=[],
        type=Path,
        help="generated CSV whose checked rows prove pairs valid (speed only)",
    )
    args = ap.parse_args(argv)
    spec = Spec(args.model_dir, args.rules_extra)
    return {
        "pairwise": cmd_pairwise,
        "oat": cmd_oat,
        "isolated": cmd_isolated,
        "accounting": cmd_accounting,
    }[args.cmd](spec, args)


if __name__ == "__main__":
    sys.exit(main())
