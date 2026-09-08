"""Follow-up to Experiment 8: does `apply_meek_rules` (Meek 1995's R1-R3
orientation-propagation rules) actually recover more of the true graph
than collider orientation alone, without ever orienting an edge wrongly
-- and does it correctly leave genuinely undetermined edges undirected,
rather than inventing an orientation the data can't support?
(docs/research-agenda.md #7e)

Two graphs, two different questions:

## Graph 1 -- a collider-then-chain, R1 propagation case

`A -> C <- B` (an unshielded collider at `C`), `C -> D -> F` (a pure
chain continuing from the collider). Collider orientation alone finds
only `A->C, B->C` -- `C->D` and `D->F` aren't part of any v-structure by
themselves. But this DAG is actually *fully identified* from its CPDAG:
Meek's R1 forces `C->D` (else `A->C<-D` would be an undetected new
collider, since `A` and `D` aren't adjacent), and then, using that
newly-derived edge, forces `D->F` the same way. This is a genuine
two-hop propagation, not a single rule application -- confirmed by hand
first in `tests/test_causal_reasoning.py` before this follow-up measures
it under sampling noise.

## Graph 2 -- a plain chain, the negative control

`A -> B -> C`, no collider anywhere. This DAG is Markov-equivalent to
`A <- B -> C` (a fork) and `A <- B <- C` (the reverse chain) -- all three
produce identical conditional-independence facts, so NO amount of
observational data can determine which one generated it. The confound
this follow-up needs to control for: collider orientation correctly
finds nothing here (no v-structure exists), but does `apply_meek_rules`
correctly leave it that way too, or does it spuriously invent an
orientation where none is actually determined? If Meek's rules oriented
anything at all on this graph, the "recovers more of the true graph"
claim from graph 1 would be worthless -- it would mean the rules orient
edges regardless of whether the data actually determines them.

Run: PYTHONPATH=. python experiments/exp08_causal_discovery/meek_rules.py
"""
from __future__ import annotations

import random
import statistics

from transintelligence.reasoning.causal import apply_meek_rules, discover_skeleton, orient_colliders

SAMPLE_SIZES = [100, 300, 1000, 3000]
SEEDS = list(range(20))
ALPHA = 0.01

PROPAGATION_TRUE_DIRECTED_EDGES = {("A", "C"), ("B", "C"), ("C", "D"), ("D", "F")}
PROPAGATION_TRUE_SKELETON = {frozenset(e) for e in PROPAGATION_TRUE_DIRECTED_EDGES}


def simulate_propagation_graph(seed: int, n: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    a, b, c, d, f = [], [], [], [], []
    for _ in range(n):
        av = rng.gauss(0, 1)
        bv = rng.gauss(0, 1)
        cv = av + bv + rng.gauss(0, 0.3)
        dv = cv + rng.gauss(0, 0.3)
        fv = dv + rng.gauss(0, 0.3)
        a.append(av); b.append(bv); c.append(cv); d.append(dv); f.append(fv)
    return {"A": a, "B": b, "C": c, "D": d, "F": f}


def simulate_plain_chain(seed: int, n: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    a, b, c = [], [], []
    for _ in range(n):
        av = rng.gauss(0, 1)
        bv = av + rng.gauss(0, 0.3)
        cv = bv + rng.gauss(0, 0.3)
        a.append(av); b.append(bv); c.append(cv)
    return {"A": a, "B": b, "C": c}


def main() -> None:
    print("=== Graph 1: collider-then-chain (4 true directed edges, all identifiable) ===")
    print(f"{'n':>6} {'collider_only_recall':>22} {'with_meek_recall':>18} {'wrong_orient_given_correct_skeleton':>37} {'skeleton_recovery_errors':>25}")
    reversed_true = {(c, p) for p, c in PROPAGATION_TRUE_DIRECTED_EDGES}
    for n in SAMPLE_SIZES:
        collider_recalls, meek_recalls = [], []
        wrong_given_correct_skeleton, skeleton_errors = 0, 0
        for seed in SEEDS:
            data = simulate_propagation_graph(seed, n)
            nodes = sorted(data.keys())
            skeleton = discover_skeleton(data, alpha=ALPHA)
            colliders = orient_colliders(skeleton, nodes)
            full = apply_meek_rules(skeleton, colliders, nodes)
            collider_recalls.append(len(colliders & PROPAGATION_TRUE_DIRECTED_EDGES) / len(PROPAGATION_TRUE_DIRECTED_EDGES))
            meek_recalls.append(len(full & PROPAGATION_TRUE_DIRECTED_EDGES) / len(PROPAGATION_TRUE_DIRECTED_EDGES))
            skeleton_correct = skeleton.undirected_edges == frozenset(PROPAGATION_TRUE_SKELETON)
            if not skeleton_correct:
                skeleton_errors += 1
            elif full & reversed_true:
                # Skeleton was right but the orientation itself was wrong --
                # this, not a skeleton error propagating downstream, is the
                # real test of apply_meek_rules' own correctness.
                wrong_given_correct_skeleton += 1
        print(f"{n:>6} {statistics.mean(collider_recalls):>22.3f} {statistics.mean(meek_recalls):>18.3f} {f'{wrong_given_correct_skeleton}/{len(SEEDS)}':>37} {f'{skeleton_errors}/{len(SEEDS)}':>25}")

    print("\n=== Graph 2: plain chain, no collider (0 true directed edges -- Markov-equivalent to a fork and a reverse chain) ===")
    print(f"{'n':>6} {'collider_only_edges_oriented':>29} {'with_meek_edges_oriented':>25}")
    for n in SAMPLE_SIZES:
        collider_counts, meek_counts = [], []
        for seed in SEEDS:
            data = simulate_plain_chain(seed, n)
            nodes = sorted(data.keys())
            skeleton = discover_skeleton(data, alpha=ALPHA)
            colliders = orient_colliders(skeleton, nodes)
            full = apply_meek_rules(skeleton, colliders, nodes)
            collider_counts.append(len(colliders))
            meek_counts.append(len(full))
        print(f"{n:>6} {statistics.mean(collider_counts):>29.3f} {statistics.mean(meek_counts):>25.3f}")


if __name__ == "__main__":
    main()
