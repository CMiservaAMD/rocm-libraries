#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare two SdpaFwdPlan bundle trees, for example a committed set and a rebuilt set.

Each root holds <tier>/SdpaFwdPlan/<Topology>/{graph.template.json,sweep.json,support.json},
for example dnn-providers/integration-tests/integration-test-bundles or the --work-dir of
make_bundles.py.

Reports the file lists, the files that are byte-identical, and the files that are identical
once every "generator_version": "<hex>" value is replaced by a placeholder. Also reports the
generator_version values seen on each side. Exit 0 only when every file is equal after that
replacement.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

GV = re.compile(rb'"generator_version": "([0-9a-f]+)"')
GV_PLACEHOLDER = b'"generator_version": "X"'
TIERS = ("quick", "standard", "comprehensive", "full")
OP_DIR = "SdpaFwdPlan"


def files(root):
    out = {}
    for tier in TIERS:
        base = root / tier / OP_DIR
        if base.is_dir():
            for f in base.rglob("*"):
                if f.is_file():
                    out[f.relative_to(root).as_posix()] = f
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "committed",
        type=Path,
        help="root of the committed tree (holds <tier>/SdpaFwdPlan)",
    )
    ap.add_argument(
        "rebuilt", type=Path, help="root of the rebuilt tree (holds <tier>/SdpaFwdPlan)"
    )
    ap.add_argument(
        "--report", type=Path, help="also write the full report to this file"
    )
    args = ap.parse_args()

    a, b = files(args.committed), files(args.rebuilt)
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    raw_same = norm_same = 0
    norm_diff = []
    gv_a, gv_b = Counter(), Counter()
    for rel in sorted(set(a) & set(b)):
        x, y = a[rel].read_bytes(), b[rel].read_bytes()
        gv_a.update(m.decode() for m in GV.findall(x))
        gv_b.update(m.decode() for m in GV.findall(y))
        if x == y:
            raw_same += 1
            norm_same += 1
        elif GV.sub(GV_PLACEHOLDER, x) == GV.sub(GV_PLACEHOLDER, y):
            norm_same += 1
        else:
            norm_diff.append(rel)
    lines = [
        f"files committed={len(a)} rebuilt={len(b)} only_committed={len(only_a)}"
        f" only_rebuilt={len(only_b)}",
        f"raw identical={raw_same} identical_after_generator_version_normalisation={norm_same}"
        f" different={len(norm_diff)}",
        f"generator_version committed={dict(gv_a)}",
        f"generator_version rebuilt={dict(gv_b)}",
    ]
    lines += [f"ONLY_COMMITTED {r}" for r in only_a[:50]]
    lines += [f"ONLY_REBUILT {r}" for r in only_b[:50]]
    lines += [f"DIFFERENT {r}" for r in norm_diff[:100]]
    ok = not only_a and not only_b and not norm_diff and len(a) > 0
    lines.append("RESULT " + ("MATCH" if ok else "MISMATCH"))
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print("\n".join(lines[:4] + lines[-1:]))
    else:
        print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
