"""Aggregate per-task rows into a results table with paired bootstrap CIs."""

from __future__ import annotations

import random
from collections import Counter, defaultdict


def summarize(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    out = {}
    for pol, rs in by.items():
        n = len(rs)
        vis = sum(r["visible_pass"] for r in rs)
        hid = sum(r["hidden_pass"] for r in rs)
        overfit = sum(r["visible_pass"] and not r["hidden_pass"] for r in rs)
        out[pol] = {
            "n": n,
            "hidden_pass": hid / n,
            "visible_pass": vis / n,
            "first_try_hidden_pass": sum(r["first_hidden_pass"] for r in rs) / n,
            "overfit_rate": overfit / max(vis, 1),
            "mean_generations": sum(r["generations"] for r in rs) / n,
            "repeat_rate": sum(r.get("repeats", 0) for r in rs)
            / max(sum(r["generations"] - 1 for r in rs), 1),
            "mean_new_tokens": sum(r["new_tokens"] for r in rs) / n,
            "first_failure": dict(Counter(r["first_failure"] for r in rs)),
        }
    return out


def paired_bootstrap(
    rows: list[dict], a: str, b: str, metric: str = "hidden_pass", n_boot: int = 2000, seed: int = 0
) -> dict:
    """Difference a - b on the same tasks, with a 95% percentile CI."""
    xa = {r["task_id"]: float(r[metric]) for r in rows if r["policy"] == a}
    xb = {r["task_id"]: float(r[metric]) for r in rows if r["policy"] == b}
    ids = sorted(set(xa) & set(xb))
    if not ids:
        raise ValueError("no shared tasks")
    diffs = [xa[i] - xb[i] for i in ids]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "a": a,
        "b": b,
        "diff": sum(diffs) / n,
        "lo": boots[int(0.025 * n_boot)],
        "hi": boots[int(0.975 * n_boot) - 1],
        "n": n,
    }
