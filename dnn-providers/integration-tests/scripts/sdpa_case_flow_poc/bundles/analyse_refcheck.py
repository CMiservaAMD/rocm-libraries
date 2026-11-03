#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Analyse reference-check output of hipdnn_sdpa_case_flow_runner.

The reference check runs the runner over the selected rows with an engine that declines
every SDPA graph, so only the GPU and CPU references run, and compares them.

Per arch and tier it reports:
  - rows expected, rows with a result, rows missing from the output;
  - GPU and CPU reference status counts; every refusal or error, with its message;
  - gpu-ref-vs-cpu-ref `ref_agree` failures, with their element statistics;
  - the fp8 1-ULP share (pooled, maximum per row, rows flagged above 1 %);
  - the maximum absolute and relative error per output dtype and tensor.
Every listed finding names its GTest case through placements.csv (case_key -> gtest_name).

placements.csv is written by make_bundles.py, or by --make-placements from bundle trees
(case metadata case_key, tier, sweep dir, case id). Rows not in any tree are reported as
"(no bundle case)".

Usage:
  analyse_refcheck.py --make-placements --placements OUT.csv [--tree TREE ...]
  analyse_refcheck.py --rows selected.csv --placements placements.csv \
      --results 'refcheck/*/rows.jsonl' [--archs gfx942,gfx950] [--json OUT.json]
"""
import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
DEFAULT_TREE = Path(__file__).resolve().parents[3] / "integration-test-bundles"
OP = "SdpaFwdPlan"
TIERS = ["quick", "standard", "comprehensive", "full"]
ONE_ULP_FLAG = 0.01

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_placements(out, trees):
    from make_bundles import gtest_name  # noqa: E402

    seen = {}
    for tree in trees:
        for tier in TIERS:
            op_dir = tree / tier / OP
            if not op_dir.is_dir():
                continue
            for sweep in sorted(p for p in op_dir.iterdir() if p.is_dir()):
                for c in json.loads((sweep / "sweep.json").read_text())["cases"]:
                    key = c.get("metadata", {}).get("case_key")
                    if key and key not in seen:
                        seen[key] = (
                            key,
                            tier,
                            sweep.name,
                            c["id"],
                            f"{tier}/{OP}/{sweep.name}",
                            gtest_name(tier, OP, sweep.name, c["id"]),
                        )
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            ["case_key", "tier", "topology", "case_id", "bundle_path", "gtest_name"]
        )
        w.writerows(sorted(seen.values()))
    print(f"placements: {len(seen)} cases -> {out}")


def load_results(patterns):
    """(arch, case_key) -> last case record seen, plus the files read."""
    out, files = {}, []
    for pat in patterns:
        for f in sorted(Path(x) for x in glob.glob(pat)):
            files.append(f)
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("record") == "case":
                    out[(r.get("arch"), r.get("case_key"))] = r
    return out, files


def ref_comparison(r):
    return next(
        (
            c
            for c in r.get("comparisons", [])
            if c.get("against") == "gpu-ref-vs-cpu-ref"
        ),
        None,
    )


def tensor_dtype(label, row):
    name = label.split("::")[-1]
    return (
        (row.get("dtype_o", "?") if name == "O" else row.get("dtype_stats", "?") or "?")
        + " "
        + name
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--make-placements", action="store_true")
    ap.add_argument(
        "--tree",
        type=Path,
        action="append",
        help="bundle tree for --make-placements (default: the repository tree); repeatable",
    )
    ap.add_argument("--placements", type=Path, required=True)
    ap.add_argument(
        "--rows", type=Path, help="expected rows (CSV), for example selected.csv"
    )
    ap.add_argument(
        "--results",
        action="append",
        help="glob of runner output JSONL files (repeatable)",
    )
    ap.add_argument("--archs", default="gfx942,gfx950")
    ap.add_argument("--json", type=Path, help="also write the findings as JSON")
    ap.add_argument(
        "--max-list", type=int, default=50, help="list at most N findings per kind"
    )
    a = ap.parse_args()
    if a.make_placements:
        make_placements(a.placements, a.tree or [DEFAULT_TREE])
        return 0
    if not a.rows or not a.results:
        ap.error("--rows and --results are required unless --make-placements is given")
    with open(a.rows, newline="") as f:
        expected = {r["case_key"]: r for r in csv.DictReader(f)}
    place = {}
    if a.placements.is_file():
        with open(a.placements, newline="") as f:
            place = {r["case_key"]: r["gtest_name"] for r in csv.DictReader(f)}
    else:
        print(
            f"warning: {a.placements} missing; run --make-placements", file=sys.stderr
        )
    results, files = load_results(a.results)
    print(
        f"result files: {len(files)}; case records: {len(results)}; expected rows: {len(expected)}"
    )

    def name(key):
        row = expected.get(key, {})
        return f"{place.get(key, '(no bundle case)')} [{row.get('case_id', '?')} {key}]"

    findings = {
        "missing": [],
        "refusals": [],
        "disagree": [],
        "fp8_flags": [],
        "not_done": [],
    }
    exit_code = 0
    for arch in a.archs.split(","):
        print(f"\n===== {arch}")
        for tier in TIERS:
            keys = [k for k, r in expected.items() if r["level"] == tier]
            if not keys:
                continue
            got = [k for k in keys if (arch, k) in results]
            missing = [k for k in keys if (arch, k) not in results]
            gpu, cpu, agree = Counter(), Counter(), Counter()
            ulp_one = ulp_cmp = 0
            ulp_max_share, flagged = 0.0, 0
            maxerr = defaultdict(lambda: [0.0, 0.0, 0])
            lists = defaultdict(list)
            for k in missing:
                lists["missing"].append(name(k))
            for k in got:
                r, row = results[(arch, k)], expected[k]
                gpu[r.get("gpu_ref_status", "-")] += 1
                cpu[r.get("cpu_ref_status", "-")] += 1
                if r.get("stage") != "done":
                    lists["not_done"].append(
                        f"{name(k)} stage={r.get('stage')} outcome={r.get('outcome')} {r.get('message', '')[:200]}"
                    )
                for ref in ("gpu", "cpu"):
                    st = r.get(f"{ref}_ref_status")
                    if st not in (None, "ok"):
                        lists["refusals"].append(
                            f"{name(k)} {ref}_ref={st}: {str(r.get(f'{ref}_ref_message', ''))[:200]}"
                        )
                c = ref_comparison(r)
                if c is None:
                    agree["no-comparison"] += 1
                    continue
                if not c.get("ran"):
                    agree["compare-error"] += 1
                    lists["disagree"].append(
                        f"{name(k)} compare error: {c.get('error', '')[:200]}"
                    )
                    continue
                agree["agree" if c.get("ref_agree") else "DISAGREE"] += 1
                stats = []
                for label, e in c.get("elements", {}).items():
                    if "max_abs" not in e:
                        continue
                    m = maxerr[tensor_dtype(label, row)]
                    m[0], m[1], m[2] = (
                        max(m[0], e["max_abs"]),
                        max(m[1], e["max_rel"]),
                        m[2] + 1,
                    )
                    if e.get("fp8"):
                        ulp_one += e["ulp_one"]
                        ulp_cmp += e["compared"]
                        ulp_max_share = max(ulp_max_share, e["ulp_one_share"])
                        flagged += e["ulp_one_share"] > ONE_ULP_FLAG
                    stats.append(
                        f"{label.split('::')[-1]}: abs={e['max_abs']:.3g} rel={e['max_rel']:.3g}"
                        + (
                            f" ulp_max={e['ulp_max']} >1ulp={e['ulp_over_one']} 1ulp_share={e['ulp_one_share']:.2e}"
                            if e.get("fp8")
                            else ""
                        )
                        + (
                            f" nonfinite_pos={e['nonfinite_position_mismatches']}"
                            if e.get("nonfinite_position_mismatches")
                            else ""
                        )
                    )
                if not c.get("ref_agree"):
                    lists["disagree"].append(
                        f"{name(k)} suite_mismatched={c.get('mismatched')} "
                        + "; ".join(stats)
                    )
                if c.get("ref_flags"):
                    lists["fp8_flags"].append(
                        f"{name(k)} {c['ref_flags']} " + "; ".join(stats)
                    )
            print(
                f"\n--- {arch} {tier}: expected {len(keys)}, with result {len(got)}, missing {len(missing)}"
            )
            print(f"  gpu_ref: {dict(gpu)}")
            print(f"  cpu_ref: {dict(cpu)}")
            print(f"  gpu-vs-cpu: {dict(agree)}")
            if ulp_cmp:
                print(
                    f"  fp8 1-ULP share: pooled {ulp_one / ulp_cmp:.2e} ({ulp_one}/{ulp_cmp}), max per row {ulp_max_share:.2e}, rows above 1 %: {flagged}"
                )
            for dt in sorted(maxerr):
                m = maxerr[dt]
                print(
                    f"  max error {dt:24} abs={m[0]:.3g} rel={m[1]:.3g} (tensors {m[2]})"
                )
            for kind in ("missing", "not_done", "refusals", "disagree", "fp8_flags"):
                items = lists[kind]
                if not items:
                    continue
                if kind in ("disagree", "refusals", "not_done"):
                    exit_code = 1
                print(f"  {kind} ({len(items)}):")
                for s in items[: a.max_list]:
                    print(f"    {s}")
                if len(items) > a.max_list:
                    print(f"    ... {len(items) - a.max_list} more")
                findings[kind] += [f"{arch} {tier} {s}" for s in items]
    if a.json:
        a.json.write_text(json.dumps(findings, indent=1))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
