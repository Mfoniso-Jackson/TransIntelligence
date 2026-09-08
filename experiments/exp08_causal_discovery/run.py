"""Experiment 8 -- can PC-style constraint-based discovery
(`discover_skeleton`/`orient_colliders`, docs/related-work.md #3c) recover
graph structure from data alone, at realistic sample sizes and with a
genuine negative control? (docs/research-agenda.md #7e)

Three ground-truth structures, three different questions:

1. **Confounding graph** (reused from experiment 6's Z->X, Z->Y, X->Y,
   X->W, Y->W, but with TRUE_EFFECT=0.5 instead of experiment 6's 0.0 --
   see "why TRUE_EFFECT differs from experiment 6" below): can the
   skeleton (undirected edge presence) be recovered as sample size grows?
   This graph's W is a **shielded** collider (X-Y is also a direct edge),
   so it is a negative case for orientation, not a positive one -- the
   test here is that discovery correctly leaves it unoriented, not that
   it finds it.
2. **Dedicated unshielded-collider graph** (A->B<-C, A and C independent
   causes of B, no A-C edge): the positive case for collider orientation
   that experiment 6's graph cannot provide.
3. **Fully independent variables** (no edges at all): the confound
   control that could kill this experiment's positive results -- does
   the algorithm just find edges everywhere given enough data, the way a
   naive multiple-testing procedure would? If the false-edge rate grows
   with N instead of staying near alpha, the "recovers structure" claim
   from (1) and (2) would be worthless.

## Why TRUE_EFFECT differs from experiment 6

Experiment 6 deliberately set X's true effect on Y to exactly 0.0, to
demonstrate confounding bias in an *effect estimate* even when the
*edge* X->Y is nominally in the graph. But skeleton discovery is a purely
statistical procedure: a structural equation term multiplied by a true
coefficient of exactly 0.0 produces no actual statistical dependence, so
X and Y would be (correctly) found conditionally independent given Z --
the X-Y edge would vanish from the discovered skeleton, changing the
ground-truth graph out from under this experiment. Reusing experiment 6's
graph *shape* requires a nonzero coefficient (0.5, matching experiment
7's convention) so the X-Y edge is real and discoverable.

Run: PYTHONPATH=. python experiments/exp08_causal_discovery/run.py
"""
from __future__ import annotations

import random
import statistics

from transintelligence.reasoning.causal import discover_skeleton, orient_colliders

# --- Graph 1: confounding structure, reused shape from experiment 6 ---
ALPHA_ZX = 0.8   # Z -> X
BETA_ZY = 0.8    # Z -> Y
TRUE_EFFECT_XY = 0.5  # X -> Y (nonzero here, unlike experiment 6 -- see module docstring)
D_XW = 0.5       # X -> W
E_YW = 0.5       # Y -> W
NOISE_SIGMA = 0.3

CONFOUNDING_TRUE_EDGES = {
    frozenset({"Z", "X"}), frozenset({"Z", "Y"}), frozenset({"X", "Y"}),
    frozenset({"X", "W"}), frozenset({"Y", "W"}),
}

# --- Graph 2: dedicated unshielded collider, A and C independent ---
A_TO_B = 0.7
C_TO_B = 0.7

COLLIDER_TRUE_EDGES = {frozenset({"A", "B"}), frozenset({"B", "C"})}

SAMPLE_SIZES = [100, 300, 1000, 3000]
SEEDS = list(range(20))
ALPHA = 0.01


def simulate_confounding(seed: int, n: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    z, x, y, w = [], [], [], []
    for _ in range(n):
        zv = rng.gauss(0, 1)
        xv = ALPHA_ZX * zv + rng.gauss(0, NOISE_SIGMA)
        yv = BETA_ZY * zv + TRUE_EFFECT_XY * xv + rng.gauss(0, NOISE_SIGMA)
        wv = D_XW * xv + E_YW * yv + rng.gauss(0, NOISE_SIGMA)
        z.append(zv); x.append(xv); y.append(yv); w.append(wv)
    return {"Z": z, "X": x, "Y": y, "W": w}


def simulate_collider(seed: int, n: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    a, b, c = [], [], []
    for _ in range(n):
        av = rng.gauss(0, 1)
        cv = rng.gauss(0, 1)
        bv = A_TO_B * av + C_TO_B * cv + rng.gauss(0, NOISE_SIGMA)
        a.append(av); b.append(bv); c.append(cv)
    return {"A": a, "B": b, "C": c}


def simulate_independent(seed: int, n: int, names: tuple[str, ...] = ("P", "Q", "R", "S")) -> dict[str, list[float]]:
    rng = random.Random(seed)
    return {name: [rng.gauss(0, 1) for _ in range(n)] for name in names}


def precision_recall(discovered: frozenset, true_edges: set) -> tuple[float, float]:
    tp = len(discovered & true_edges)
    fp = len(discovered - true_edges)
    fn = len(true_edges - discovered)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


def main() -> None:
    print("=== Skeleton recovery: confounding graph (5 true edges, W is a SHIELDED collider) ===")
    print(f"{'n':>6} {'mean_precision':>15} {'mean_recall':>13} {'w_falsely_oriented':>20}")
    for n in SAMPLE_SIZES:
        precisions, recalls, false_orientations = [], [], 0
        for seed in SEEDS:
            data = simulate_confounding(seed, n)
            skeleton = discover_skeleton(data, alpha=ALPHA)
            p, r = precision_recall(skeleton.undirected_edges, CONFOUNDING_TRUE_EDGES)
            precisions.append(p); recalls.append(r)
            directed = orient_colliders(skeleton, sorted(data.keys()))
            if ("X", "W") in directed or ("Y", "W") in directed:
                false_orientations += 1
        print(f"{n:>6} {statistics.mean(precisions):>15.3f} {statistics.mean(recalls):>13.3f} {f'{false_orientations}/{len(SEEDS)}':>20}")

    print("\n=== Collider orientation: dedicated unshielded graph A->B<-C (2 true edges) ===")
    print(f"{'n':>6} {'mean_precision':>15} {'mean_recall':>13} {'correctly_oriented':>20}")
    for n in SAMPLE_SIZES:
        precisions, recalls, correct_orientations = [], [], 0
        for seed in SEEDS:
            data = simulate_collider(seed, n)
            skeleton = discover_skeleton(data, alpha=ALPHA)
            p, r = precision_recall(skeleton.undirected_edges, COLLIDER_TRUE_EDGES)
            precisions.append(p); recalls.append(r)
            directed = orient_colliders(skeleton, sorted(data.keys()))
            if directed == frozenset({("A", "B"), ("C", "B")}):
                correct_orientations += 1
        print(f"{n:>6} {statistics.mean(precisions):>15.3f} {statistics.mean(recalls):>13.3f} {f'{correct_orientations}/{len(SEEDS)}':>20}")

    print("\n=== Negative control: 4 mutually independent variables (0 true edges) ===")
    print(f"{'n':>6} {'trials_with_a_false_edge':>26} {'mean_false_edges_per_trial':>28}")
    for n in SAMPLE_SIZES:
        trials_with_edge, edge_counts = 0, []
        for seed in SEEDS:
            data = simulate_independent(seed, n)
            skeleton = discover_skeleton(data, alpha=ALPHA)
            edge_counts.append(len(skeleton.undirected_edges))
            if skeleton.undirected_edges:
                trials_with_edge += 1
        print(f"{n:>6} {f'{trials_with_edge}/{len(SEEDS)}':>26} {statistics.mean(edge_counts):>28.3f}")


if __name__ == "__main__":
    main()
