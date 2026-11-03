# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The generation modes. Each mode returns a list of (meta, row) tuples."""

from __future__ import annotations

import random

from .model import Model
from .pairs import axis_pos, make_pair
from .solver import Solver


def _fmt(v):
    return "-" if v is None else v


def _diff(model: Model, base: dict, row: dict, skip=()) -> str:
    items = [
        f"{a}={_fmt(row[a])}" for a in model.axes if a not in skip and row[a] != base[a]
    ]
    return ";".join(items)


def one_at_a_time(model: Model, solver: Solver, base_names) -> tuple:
    """Vary one value per row from each base. Return rows and the values no valid row can hold."""
    rows, excluded = [], []
    for name in base_names:
        base = model.base_cases[name]
        rows.append(({"base": name, "varied": "-", "also_changed": ""}, dict(base)))
        for a in model.axes:
            for v in model.values[a]:
                if v == base[a]:
                    continue
                if not solver.feasible({a: v}):
                    excluded.append((name, a, v))
                    continue
                row = solver.complete({a: v}, prefs=base)
                meta = {
                    "base": name,
                    "varied": f"{a}={v}",
                    "also_changed": _diff(model, base, row, skip=(a,)),
                }
                rows.append((meta, row))
    return rows, excluded


def pairwise(
    model: Model,
    solver: Solver,
    valid_pairs,
    rng: random.Random,
    fixed=None,
    candidates: int = 2,
) -> list:
    """Greedy covering set: every pair in `valid_pairs` appears in at least one row.

    Each row starts from the value that appears in the most uncovered pairs and one uncovered
    partner. The other axes follow in a seeded random order; each takes the value that keeps the
    row completable and covers the most uncovered pairs with the values chosen so far.
    """
    fixed = dict(fixed or {})
    pos = axis_pos(model)
    uncovered = set(valid_pairs)
    partners = {}
    for f, g in sorted(uncovered):
        partners.setdefault(f, set()).add(g)
        partners.setdefault(g, set()).add(f)
    free_axes = [a for a in model.axes if a not in fixed]
    rows = []

    def build_row():
        top = max(len(s) for s in partners.values() if s)
        seeds = sorted(f for f, s in partners.items() if len(s) == top)
        f = rng.choice(seeds)
        g = rng.choice(sorted(partners[f]))
        row = dict(fixed)
        row[f[0]] = f[1]
        row[g[0]] = g[1]
        order = [a for a in free_axes if a not in row]
        rng.shuffle(order)
        for a in order:
            best, best_score = [], -1
            for v in solver.domain(a):
                trial = dict(row)
                trial[a] = v
                if not solver.feasible(trial):
                    continue
                score = 0
                if v is not None:
                    fv = (a, v)
                    for b, w in row.items():
                        if w is not None and make_pair(pos, fv, (b, w)) in uncovered:
                            score += 1
                if score > best_score:
                    best, best_score = [v], score
                elif score == best_score:
                    best.append(v)
            if not best:
                raise AssertionError(f"no completable value for {a}")
            row[a] = best[0] if len(best) == 1 else rng.choice(best)
        return {a: row[a] for a in model.axes}

    def covered_by(row):
        feats = [(a, row[a]) for a in model.axes if row[a] is not None]
        out = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                p = (feats[i], feats[j])
                if p in uncovered:
                    out.append(p)
        return out

    while uncovered:
        best_row, best_cov = None, None
        for _ in range(candidates):
            r = build_row()
            cov = covered_by(r)
            if best_cov is None or len(cov) > len(best_cov):
                best_row, best_cov = r, cov
        for f, g in best_cov:
            uncovered.discard((f, g))
            partners[f].discard(g)
            partners[g].discard(f)
        rows.append(({}, best_row))
    return rows
