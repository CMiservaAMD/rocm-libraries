#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rule checker: no row of a generated CSV breaks a rule, and child axes appear only under their parent.

Exit status 0 when every row of every file passes, 1 otherwise. Each problem names the case id
and the rule id or the axis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_common import Spec, read_cases  # noqa: E402

MAX_LISTED = 20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--rules-extra", action="append", default=[], type=Path)
    ap.add_argument("csv", nargs="+", type=Path)
    args = ap.parse_args(argv)

    spec = Spec(args.model_dir, args.rules_extra)
    failed = False
    for path in args.csv:
        n, bad = 0, []
        for case_id, _, row in read_cases(path, spec):
            n += 1
            for p in spec.row_problems(row):
                bad.append(f"{case_id}: {p}")
        status = "PASS" if not bad else "FAIL"
        print(f"{status} rules {path.name}: {n} rows, {len(bad)} problems")
        for line in bad[:MAX_LISTED]:
            print(f"  {line}")
        if len(bad) > MAX_LISTED:
            print(f"  ... {len(bad) - MAX_LISTED} more")
        failed |= bool(bad) or n == 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
