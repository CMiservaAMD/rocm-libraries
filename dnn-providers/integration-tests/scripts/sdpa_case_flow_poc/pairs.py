# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Value pairs: which pairs can appear in a row.

A feature is (axis, value). A pair is two features of different axes, ordered by axis position
in model.yaml. A pair is valid when at least one row that breaks no rule contains both features.
"""

from __future__ import annotations

import itertools

from .model import Model
from .solver import Solver


def axis_pos(model: Model) -> dict:
    return {a: i for i, a in enumerate(model.axes)}


def make_pair(pos: dict, f, g):
    return (f, g) if pos[f[0]] < pos[g[0]] else (g, f)


def candidate_pairs(model: Model, skip_axes=()) -> list:
    pos = axis_pos(model)
    feats = [(a, v) for a in model.axes if a not in skip_axes for v in model.values[a]]
    out = []
    for f, g in itertools.combinations(feats, 2):
        if f[0] != g[0]:
            out.append(make_pair(pos, f, g))
    return out


def feasible_pairs(solver: Solver, pairs, fixed=None) -> list:
    fixed = dict(fixed or {})
    single = {}

    def single_ok(f):
        if f not in single:
            single[f] = solver.feasible({**fixed, f[0]: f[1]})
        return single[f]

    out = []
    for f, g in pairs:
        if not single_ok(f) or not single_ok(g):
            continue
        if solver.feasible({**fixed, f[0]: f[1], g[0]: g[1]}):
            out.append((f, g))
    return out
