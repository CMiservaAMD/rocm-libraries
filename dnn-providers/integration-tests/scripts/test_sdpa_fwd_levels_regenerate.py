#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regenerate the SDPA forward case CSV from the committed model files and check its SHA-256.

The generator (sdpa_case_flow_poc.levels) reads test-plans/sdpa-fwd/ and writes levels.csv.
The expected SHA-256 is in test-plans/sdpa-fwd/levels.csv.sha256. A change to the model files,
the timing files or the generator that changes the CSV fails this test. Update the SHA-256 file
in the same change.

Usage::

    python3 test_sdpa_fwd_levels_regenerate.py --work-dir DIR

The CSV is written to DIR, which is emptied first. Exit status 77 means PyYAML is missing.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_DIR = SCRIPT_DIR.parent / "test-plans" / "sdpa-fwd"
EXPECTED_FILE = PLAN_DIR / "levels.csv.sha256"
SEED = 2893
TIMINGS = (("gfx942", "timings-gfx942.csv"), ("gfx950", "timings-gfx950.csv"))
SKIP = 77


def expected_sha256() -> str:
    return EXPECTED_FILE.read_text(encoding="utf-8").split()[0]


def regenerate(work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    cmd = [
        sys.executable,
        "-m",
        "sdpa_case_flow_poc.levels",
        "--model-dir",
        str(PLAN_DIR),
        "--out-dir",
        str(work_dir),
        "--seed",
        str(SEED),
    ]
    for arch, name in TIMINGS:
        cmd += ["--timings", f"{arch}={PLAN_DIR / name}"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    return work_dir / "levels.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    try:
        import yaml  # noqa: F401
    except ImportError:
        print("SKIP: PyYAML is not installed")
        return SKIP

    csv_path = regenerate(args.work_dir.resolve())
    got = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    want = expected_sha256()
    if got != want:
        print(f"FAIL: {csv_path} has SHA-256 {got}")
        print(f"      {EXPECTED_FILE} expects {want}")
        print(
            "      If the change to the CSV is intended, write the new SHA-256 to that file."
        )
        return 1
    print(f"PASS: levels.csv SHA-256 {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
