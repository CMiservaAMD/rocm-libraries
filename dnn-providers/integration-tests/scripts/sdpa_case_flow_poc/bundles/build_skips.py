#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write the ASM_SDPA_ENGINE [[test_skips]] entries for the SdpaFwdPlan validation failures.

Failures come from every run in the --validate folders (analyse.py layout), default and
host input fill alike (a case that failed under either fill is a failure). A glob is used
only when, on every arch of the entry, every case it matches either was declined or failed
with the entry's tag; otherwise the entry lists exact GTest names.

The skips go at the end of the TOML, after a "SdpaFwdPlan known engine defects" comment
line. A section written earlier by this script is replaced; the text before it is kept
byte for byte. Every case in placements.csv must have a result on each arch.

Usage: build_skips.py --validate DIR [--validate DIR ...] --levels levels.csv
                      --placements placements.csv --out-dir DIR [--toml TOML] [--dry-run]
"""
import argparse
import collections
import fnmatch
import json
import os
import sys
import tomllib
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse as A  # noqa: E402

DEFAULT_TOML = (
    Path(__file__).resolve().parents[4]
    / "hip-kernel-provider"
    / "config"
    / "ASM_SDPA_ENGINE.toml"
)
ARCHS = ["gfx942", "gfx950"]

COMMENT = {
    "T11": "Non-fp8 SdpaFwdPlan cases with a Q, K or V descale. The reference gap is ALMIOPEN-3132.",
    "T5": "SdpaFwdPlan cases with FP8_E4M3_FNUZ inputs.",
    "T7": "SdpaFwdPlan cases with a negative or zero attention scale.",
    "T6": "SdpaFwdPlan cases that request the dropout RNG dump.",
    "T4": "SdpaFwdPlan cases that request the Max and Sum-exp outputs.",
    "T3": "SdpaFwdPlan cases with a Stats layout the kernel cannot address.",
    "T2": "SdpaFwdPlan cases with overlapping or zero Q/K/V sequence strides. The result depends on the input data.",
}
BEGIN = "\n# -- SdpaFwdPlan known engine defects --\n"


def classify_all(validate_dirs, levels, placements):
    _, by_name = A.load_rows(levels, placements)
    outcomes = A.collect_outcomes(validate_dirs)
    universe = sorted(by_name)
    state = (
        {}
    )  # (arch, name) -> "failed:<tag>" | "failed:None" | passed | declined | ...
    flips = []
    for arch in ARCHS:
        for name in universe:
            runs = outcomes[arch].get(name)
            if not runs:
                state[(arch, name)] = "missing"
                continue
            v = A.verdict(runs)
            if v == "failed":
                state[(arch, name)] = "failed:" + str(
                    A.primary_tag(by_name[name], arch, runs)
                )
                if any(s == "passed" for _, s, _ in runs):
                    flips.append(
                        (arch, name, [(os.path.basename(x), s) for x, s, _ in runs])
                    )
            else:
                state[(arch, name)] = v
    for name in universe:
        if all(state[(a, name)].startswith("failed:") for a in ARCHS):
            runs = [r for a in ARCHS for r in outcomes[a][name]]
            tags = {A.primary_tag(by_name[name], a, runs) for a in ARCHS}
            if len(tags) == 1:
                for a in ARCHS:
                    state[(a, name)] = f"failed:{next(iter(tags))}"
    return by_name, universe, state, flips


_GLOB_CHECKED = {}


def glob_ok(pattern, tag, archs, universe, state):
    key = (pattern, tag, archs)
    if key not in _GLOB_CHECKED:
        _GLOB_CHECKED[key] = _glob_ok(pattern, tag, archs, universe, state)
    return _GLOB_CHECKED[key]


def _glob_ok(pattern, tag, archs, universe, state):
    matched = fnmatch.filter(universe, pattern)
    hits = 0
    for arch in archs:
        for n in matched:
            s = state[(arch, n)]
            if s == f"failed:{tag}":
                hits += 1
            elif s != "declined":
                return False
    return hits > 0


def candidate_globs(name, tag):
    if tag == "T2":
        return []
    suite, case = name.split(".", 1)
    dtype = case.rsplit("_", 1)[0]
    return [f"*_SdpaFwdPlan_*.{dtype}_*", f"{suite}.*", f"{suite}.{dtype}_*"]


def build(validate_dirs, levels, placements):
    by_name, universe, state, flips = classify_all(validate_dirs, levels, placements)
    missing = sorted(k for k, v in state.items() if v == "missing")
    assert not missing, f"cases absent from the validation XML: {missing[:5]}"
    untagged = sorted(k for k, v in state.items() if v == "failed:None")
    groups = collections.defaultdict(set)
    for name in universe:
        tags = {a: state[(a, name)] for a in ARCHS}
        for tag in A.REASON:
            archs = tuple(a for a in ARCHS if tags[a] == f"failed:{tag}")
            if archs:
                groups[(tag, archs)].add(name)
    entries = []
    for tag in A.REASON:
        for archs in [("gfx942", "gfx950"), ("gfx942",), ("gfx950",)]:
            names = groups.get((tag, archs))
            if not names:
                continue
            filters = []
            left = {
                n
                for n in names
                if not any(
                    e["tag"] == tag
                    and set(archs) <= set(e["archs"])
                    and any(fnmatch.fnmatchcase(n, f) for f in e["filters"])
                    for e in entries
                )
            }
            for name in sorted(names):
                if name not in left:
                    continue
                for pat in candidate_globs(name, tag):
                    if pat in filters:
                        break
                    if glob_ok(pat, tag, archs, universe, state):
                        filters.append(pat)
                        left -= {n for n in left if fnmatch.fnmatchcase(n, pat)}
                        break
            globs = list(filters)
            filters = globs + sorted(left)
            if not filters:
                continue
            entries.append(
                {
                    "tag": tag,
                    "archs": list(archs),
                    "filters": filters,
                    "globs": globs,
                    "exact": len(left),
                    "cases": len(names),
                }
            )
    return entries, untagged, flips, state, by_name


def render(entries):
    out = [BEGIN]
    for e in entries:
        out.append(f"# {COMMENT[e['tag']]}\n")
        out.append("[[test_skips]]\n")
        out.append("archs = [" + ", ".join(f'"{a}"' for a in e["archs"]) + "]\n")
        out.append(
            "filters = [\n" + "".join(f'    "{f}",\n' for f in e["filters"]) + "]\n"
        )
        out.append(f'reason = "{A.skip_reason(e["tag"])}"\n\n')
    return "".join(out).rstrip("\n") + "\n"


def split_toml(text):
    """The TOML text before any section this script wrote earlier."""
    head, sep, _ = text.partition(BEGIN)
    if sep and "[[test_skips]]" in head:
        sys.exit(
            "build_skips: the TOML has [[test_skips]] entries before the generated section"
        )
    return head


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--validate",
        type=Path,
        action="append",
        required=True,
        help="folder of <arch>-job<id>/ validation runs; repeat for more",
    )
    ap.add_argument("--levels", type=Path, required=True)
    ap.add_argument(
        "--placements",
        type=Path,
        required=True,
        help="placements.csv from make_bundles.py",
    )
    ap.add_argument("--bundle-dir", type=Path, default=A.BUNDLES)
    ap.add_argument("--toml", type=Path, default=DEFAULT_TOML)
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="where build_skips.json and multi-tag-cases.txt go",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report the entries; do not write the TOML",
    )
    args = ap.parse_args()
    A.BUNDLES = args.bundle_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries, untagged, flips, state, by_name = build(
        args.validate, args.levels, args.placements
    )
    summary = {
        "entries": [
            {k: e[k] for k in ("tag", "archs", "cases", "globs", "exact")}
            for e in entries
        ],
        "untagged_failures": untagged,
        "fill_flips": flips,
        "t2_accepted_passing": sorted(
            (a, n)
            for (a, n), s in state.items()
            if s == "passed" and "T2" in A.tags_for(by_name[n], a)
        ),
    }
    (args.out_dir / "build_skips.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["entries"], indent=1))
    print(
        "untagged failures:",
        len(untagged),
        "fill flips:",
        len(flips),
        "T2-tagged passing:",
        len(summary["t2_accepted_passing"]),
    )
    if args.dry_run:
        return
    old = split_toml(args.toml.read_text())
    new = old + render(entries)
    parsed = tomllib.loads(new)
    assert len(parsed.get("test_skips", [])) == len(entries)
    for got, e in zip(parsed["test_skips"], entries):
        assert got["filters"] == e["filters"] and got["archs"] == e["archs"]
        assert got["reason"] == A.skip_reason(e["tag"])
        assert got["reason"].startswith("ALMIOPEN-") and "[" not in got["reason"]
    assert new.startswith(old)
    args.toml.write_text(new)
    assert args.toml.read_text() == new
    print("wrote", args.toml)
    write_multi_tag(parsed["test_skips"], state, by_name, args.out_dir)


def write_multi_tag(skips, state, by_name, out_dir):
    """List skipped failures that carry more than one keyed tag."""
    rows = collections.defaultdict(list)
    for (arch, name), s in sorted(state.items()):
        if not s.startswith("failed:"):
            continue
        tags = [t for t in A.tags_for(by_name[name], arch) if t in A.TICKET]
        if len(tags) < 2:
            continue
        e = A.find_skip(skips, name, arch)
        assert e is not None, name
        used = A.reason_tag(e["reason"])
        assert used in tags, (name, used, tags)
        others = ", ".join(f"{t} ({A.TICKET[t]})" for t in tags if t != used)
        rows[(name, used, others)].append(arch)
    lines = ["# gtest_name\tarchs\tskipped_under\tother_tags"]
    for (name, used, others), archs in sorted(rows.items()):
        lines.append(f"{name}\t{','.join(archs)}\t{used} ({A.TICKET[used]})\t{others}")
    (out_dir / "multi-tag-cases.txt").write_text("\n".join(lines) + "\n")
    print("multi-tag cases:", len(rows))


if __name__ == "__main__":
    main()
