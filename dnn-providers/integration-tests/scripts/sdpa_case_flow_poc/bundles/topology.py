#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deterministic topology (sweep folder) names for SDPA forward case rows.

A row's strict signature is the set of tensor roles of its graph plus the value form of
each by-value or runtime tensor (select_rows.Model.tensors). The name lists the optional
roles (every role except Q, K, V and O) in a fixed order, one token per role, joined by
'_'. A row with no optional role is 'Base'. A name longer than 64 characters keeps its
leading tokens and ends with '_' and 6 hex digits of the sha256 of the full name.

The command line computes the name of every row of the CSV, asserts that signatures and
names map one-to-one, and writes topologies.csv.
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_rows  # noqa: E402

MAX_LEN = 64
ALWAYS = ("q", "k", "v", "o")

# Fixed order of the optional roles and the token of each.
ROLE_TOKENS = (
    ("ragged_offset_q", "RgQ"),
    ("ragged_offset_k", "RgK"),
    ("ragged_offset_v", "RgV"),
    ("ragged_offset_o", "RgO"),
    ("seq_len_q", "SlQ"),
    ("seq_len_kv", "SlKv"),
    ("block_mask", "Bm"),
    ("bias", "Bias"),
    ("sink", "Sink"),
    ("attn_scale[by_value]", "ScBv"),
    ("attn_scale[runtime]", "ScRt"),
    ("attn_scale", "ScDev"),
    ("dropout_seed[by_value]", "DSeedBv"),
    ("dropout_offset[by_value]", "DOffBv"),
    ("dropout_mask", "DMask"),
    ("dropout_scale[by_value]", "DScBv"),
    ("rng_dump", "Rng"),
    ("stats", "Stats"),
    ("logit_max", "Max"),
    ("score_sum_exp", "SumExp"),
    ("amax_s", "AmaxS"),
    ("amax_o", "AmaxO"),
    ("descale_q[by_value]", "DqBv"),
    ("descale_q", "Dq"),
    ("descale_k[by_value]", "DkBv"),
    ("descale_k", "Dk"),
    ("descale_v[by_value]", "DvBv"),
    ("descale_v", "Dv"),
    ("descale_s[by_value]", "DsBv"),
    ("scale_s[by_value]", "SsBv"),
    ("scale_o[by_value]", "SoBv"),
    ("page_table_k", "PtK"),
    ("page_table_k_offset", "PtKOff"),
    ("page_table_v", "PtV"),
    ("page_table_v_offset", "PtVOff"),
    ("page_table_kv", "PtKv"),
    ("page_table_kv_offset", "PtKvOff"),
)
_ORDER = {role: i for i, (role, _) in enumerate(ROLE_TOKENS)}
_TOKEN = dict(ROLE_TOKENS)


def topology_name(signature):
    """The topology name of a strict signature (a tuple of tensor roles)."""
    for role in ALWAYS:
        if role not in signature:
            raise ValueError(f"signature {signature} has no {role}")
    optional = [r for r in signature if r not in ALWAYS]
    unknown = [r for r in optional if r not in _ORDER]
    if unknown:
        raise ValueError(f"no topology token for role(s) {unknown}")
    tokens = [_TOKEN[r] for r in sorted(optional, key=_ORDER.__getitem__)]
    full = "_".join(tokens) or "Base"
    if len(full) <= MAX_LEN:
        return full
    suffix = "_" + hashlib.sha256(full.encode()).hexdigest()[:6]
    head = ""
    for tok in tokens:
        cand = f"{head}_{tok}" if head else tok
        if len(cand) + len(suffix) > MAX_LEN:
            break
        head = cand
    return head + suffix


def row_topology(model, row):
    return topology_name(model.tensors(row, True))


def check_one_to_one(model, rows):
    """{signature: name} over the rows; exits when two signatures share a name."""
    by_sig = {}
    by_name = {}
    for r in rows:
        sig = model.tensors(r, True)
        name = by_sig.get(sig)
        if name is None:
            name = topology_name(sig)
            by_sig[sig] = name
        other = by_name.setdefault(name, sig)
        if other != sig:
            sys.exit(f"topology: name {name} is shared by {other} and {sig}")
        if len(name) > MAX_LEN:
            sys.exit(f"topology: name {name} is longer than {MAX_LEN}")
    return by_sig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, required=True, help="levels.csv")
    ap.add_argument(
        "--class-values", type=Path, required=True, help="class-to-value.json"
    )
    ap.add_argument("--out", type=Path, required=True, help="topologies.csv to write")
    args = ap.parse_args()

    _, _, rows = select_rows.read_rows(args.input)
    model = select_rows.Model(args.class_values)
    by_sig = check_one_to_one(model, rows)

    counts = defaultdict(Counter)
    for r in rows:
        counts[by_sig[model.tensors(r, True)]][r["level"]] += 1
    shortened = sum(
        1
        for n in by_sig.values()
        if "_" in n
        and len(n.rsplit("_", 1)[-1]) == 6
        and all(c in "0123456789abcdef" for c in n.rsplit("_", 1)[-1])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["topology", "signature"] + select_rows.LEVELS)
        for sig, name in sorted(by_sig.items(), key=lambda kv: kv[1]):
            w.writerow(
                [name, " ".join(sig)] + [counts[name][lv] for lv in select_rows.LEVELS]
            )
    print(
        f"topology: {len(rows)} rows, {len(by_sig)} signatures, {len(by_sig)} names, one-to-one;"
        f" {shortened} shortened names; wrote {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
