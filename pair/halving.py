"""Step 2 allocation: successive halving of PRE/POST continuations.

Every boundary receives one PRE and one POST continuation (draw 0). After each round the boundaries that took part
are ranked by their current evidence of harm, computed on the draws they have so far,

    score = max(mean(pass PRE - pass POST) / hazard_min, mean(steps POST - steps PRE) / burden_min),

and the top `keep` fraction receives the next PRE/POST pair, up to `rounds` pairs. A boundary without a complete pair
for every earlier round ranks last; ties are broken by a seeded random order over boundary ids. Only boundaries that
complete every round can be retained by pair.verifier, so a boundary is retained on exactly `rounds` pairs.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROUNDS, KEEP, SEED = 3, 0.5, 2027


def load_draws(path) -> dict:
    """{boundary_id: {"PRE": {draw: record}, "POST": {draw: record}}} over the successful (non-error) records; a
    boundary whose records all failed keeps an empty entry, so it still takes part (and ranks last)."""
    by = defaultdict(lambda: {"PRE": {}, "POST": {}})
    if path is not None and Path(path).exists():
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                by[r["boundary_id"]]
                if not r.get("error") and r.get("arm") in ("PRE", "POST"):
                    by[r["boundary_id"]][r["arm"]][int(r["draw"])] = r
    return by


def score(arms, n_draws: int, hazard_min: float, burden_min: float) -> float:
    """Evidence of harm on draws 0..n_draws-1; -inf when any of those pairs is incomplete."""
    if arms is None or any(d not in arms["PRE"] or d not in arms["POST"] for d in range(n_draws)):
        return float("-inf")
    pre = [arms["PRE"][d] for d in range(n_draws)]
    post = [arms["POST"][d] for d in range(n_draws)]
    hazard = np.mean([bool(r["success"]) for r in pre]) - np.mean([bool(r["success"]) for r in post])
    burden = np.mean([len(r["steps"]) for r in post]) - np.mean([len(r["steps"]) for r in pre])
    return float(max(hazard / hazard_min, burden / burden_min))


def active(boundary_ids, draws: dict, round_idx: int, keep: float = KEEP, hazard_min: float = 0.5,
           burden_min: float = 5.0, seed: int = SEED) -> set:
    """Boundaries that receive draw `round_idx` (0-based): all of them in round 0, then the top `keep` of the
    previous round's participants by score on the draws before this round."""
    ids = sorted(boundary_ids)
    tie = dict(zip(ids, np.random.default_rng(seed).random(len(ids))))
    current = ids
    for r in range(1, round_idx + 1):
        ranked = sorted(current, key=lambda b: (-score(draws.get(b), r, hazard_min, burden_min), tie[b]))
        current = ranked[:int(round(keep * len(current)))]
    return set(current)
