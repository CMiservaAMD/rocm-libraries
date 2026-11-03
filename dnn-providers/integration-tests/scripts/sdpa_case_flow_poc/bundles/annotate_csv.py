#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write the annotated case CSV: levels.csv with five columns appended.

  topology           sweep folder name of the row's strict signature (every row has one)
  bundle_status      bundle | excluded
  exclusion_reasons  ';'-joined reasons from exclusions.csv (empty for a bundle row)
  bundle_path        <tier>/<OpDir>/<Topology> (bundle rows only)
  gtest_name         <tier>_<OpDir>_<Topology>.<case id> (bundle rows only)

Bundle rows take bundle_path and gtest_name from placements.csv (make_bundles.py).
--predict computes them from the naming rules instead, for checks before any import.
Every line of the output is the input line plus the appended fields, so removing the
appended columns gives back the input bytes. The script checks that before it writes.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import make_bundles  # noqa: E402
import select_rows  # noqa: E402
import topology  # noqa: E402

APPENDED = [
    "topology",
    "bundle_status",
    "exclusion_reasons",
    "bundle_path",
    "gtest_name",
]


def strip_appended(text, n=len(APPENDED)):
    out = []
    for line in text.split("\n")[:-1]:
        out.append(line.rsplit(",", n)[0])
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, required=True, help="levels.csv")
    ap.add_argument(
        "--class-values", type=Path, required=True, help="class-to-value.json"
    )
    ap.add_argument(
        "--selection-dir", type=Path, required=True, help="select_rows.py output folder"
    )
    ap.add_argument(
        "--placements", type=Path, help="placements.csv from make_bundles.py"
    )
    ap.add_argument(
        "--predict",
        action="store_true",
        help="derive bundle paths from the naming rules",
    )
    ap.add_argument("--op-dir", default="SdpaFwdPlan")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if bool(args.placements) == bool(args.predict):
        sys.exit("annotate_csv: give exactly one of --placements and --predict")

    raw = args.input.read_bytes().decode("ascii")
    header, lines, rows = select_rows.read_rows(args.input)
    model = select_rows.Model(args.class_values)
    topology.check_one_to_one(model, rows)

    with open(args.selection_dir / "exclusions.csv", newline="") as f:
        excluded = {r["case_key"]: r["reasons"] for r in csv.DictReader(f)}
    _, _, selected = select_rows.read_rows(args.selection_dir / "selected.csv")
    selected_keys = {r["case_key"] for r in selected}
    all_keys = [r["case_key"] for r in rows]
    if len(set(all_keys)) != len(all_keys):
        sys.exit("annotate_csv: case_key is not unique in the input")
    if selected_keys & set(excluded) or selected_keys | set(excluded) != set(all_keys):
        sys.exit(
            "annotate_csv: selected.csv and exclusions.csv do not partition the input rows"
        )

    placements = {}
    if args.placements:
        with open(args.placements, newline="") as f:
            placements = {r["case_key"]: r for r in csv.DictReader(f)}
        if set(placements) != selected_keys:
            sys.exit(
                f"annotate_csv: placements cover {len(placements)} rows; {len(selected_keys)} are selected"
                f" (missing {sorted(selected_keys - set(placements))[:5]},"
                f" extra {sorted(set(placements) - selected_keys)[:5]})"
            )

    out_lines = [header + "," + ",".join(APPENDED)]
    for line, r in zip(lines, rows):
        topo = topology.row_topology(model, r)
        key = r["case_key"]
        if key in selected_keys:
            if args.placements:
                p = placements[key]
                if p["topology"] != topo or p["tier"] != r["level"]:
                    sys.exit(f"annotate_csv: placement of {key} disagrees with the row")
                path, gtest = p["bundle_path"], p["gtest_name"]
            else:
                cid = make_bundles.case_id(r)
                path = f"{r['level']}/{args.op_dir}/{topo}"
                gtest = make_bundles.gtest_name(r["level"], args.op_dir, topo, cid)
            fields = [topo, "bundle", "", path, gtest]
        else:
            fields = [topo, "excluded", excluded[key], "", ""]
        if any("," in x or '"' in x or "\n" in x for x in fields):
            sys.exit(f"annotate_csv: a field of {key} needs quoting: {fields}")
        out_lines.append(line + "," + ",".join(fields))
    text = "\n".join(out_lines) + "\n"

    if strip_appended(text) != raw:
        sys.exit(
            "annotate_csv: removing the appended columns does not give back the input"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(text)
    if strip_appended(tmp.read_text()) != raw:
        sys.exit("annotate_csv: the written file does not round-trip")
    tmp.replace(args.out)
    n_bundle = len(selected_keys)
    print(
        f"annotate_csv: {len(rows)} rows ({n_bundle} bundle, {len(rows) - n_bundle} excluded);"
        f" round-trip to the input bytes holds; wrote {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
