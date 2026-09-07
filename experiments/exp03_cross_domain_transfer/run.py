"""Experiment 3 -- minimal cross-domain structural transfer
(docs/research-agenda.md #7).

Trains a LearnedEmbeddingAgent's slots on DOMAIN_A (finance-shaped), then
warm-starts a fresh agent's slots from those trained values before running
it on DOMAIN_B (knowledge-shaped). Compares against an agent trained on
DOMAIN_B from scratch (random init), on the *same* DOMAIN_B environment
realization (paired by seed).

RFAwareAgent is not used here: its "knowledge" is just its ReferenceFrame
list, which is trivial to hand over verbatim if the target domain uses the
same numeric frames -- there is nothing *learned* to test transfer of.
LearnedEmbeddingAgent's (w, b) slot values are the only thing in this
codebase that are genuinely learned from data and could meaningfully
transfer or fail to.

Two conditions:
  - DOMAIN_B_ISOMORPHIC: same (baseline, direction) numeric structure as
    DOMAIN_A, different surface labels. Per Gentner (1983), this is a real
    analogy -- shared relations, different attributes.
  - DOMAIN_B_NONISOMORPHIC (control): same labels as the isomorphic
    version, but a genuinely different numeric structure. If "transfer"
    survives this swap, it wasn't testing structure.

Run: PYTHONPATH=. python experiments/exp03_cross_domain_transfer/run.py
"""
from __future__ import annotations

import random
import statistics

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.agents import LearnedEmbeddingAgent
from experiments.exp03_cross_domain_transfer.domains import (
    DOMAIN_A_FRAMES,
    DOMAIN_B_ISOMORPHIC_FRAMES,
    DOMAIN_B_NONISOMORPHIC_FRAMES,
)

N_STEPS_A = 3000
N_STEPS_B = 3000
SWITCH_PERIOD = 40
JITTER = 10
NOISE_SIGMA = 0.05
EARLY_WINDOW = 300  # first ~7-8 switch cycles: the sample-efficiency window
TRIALS = list(range(10))


def random_differentiated_experts(n_slots: int, seed: int, scale: float = 3.0) -> list[tuple[float, float]]:
    """A control for the transfer condition: slots initialized already
    differentiated at roughly the scale trained experts converge to
    (~1-8 in practice), but with no domain-A training at all -- isolates
    "any pre-differentiated start helps" from "the transferred values are
    actually informed by domain A's rules"."""
    rng = random.Random(seed)
    return [(rng.gauss(0, scale), rng.gauss(0, scale)) for _ in range(n_slots)]


def train_on_domain_a(seed: int) -> list[tuple[float, float]]:
    env = FrameSwitchEnv(DOMAIN_A_FRAMES, switch_period=SWITCH_PERIOD, jitter=JITTER, noise_sigma=NOISE_SIGMA, seed=seed)
    agent = LearnedEmbeddingAgent(len(DOMAIN_A_FRAMES), seed=seed)
    for _ in range(N_STEPS_A):
        info = env.observe()
        reward = env.feedback(agent.act(info.raw))
        agent.update(reward)
    return list(agent.experts)


def run_on_domain_b(frames: list, seed: int, initial_experts: list[tuple[float, float]] | None) -> tuple[float, float]:
    """Returns (early_window_accuracy, overall_accuracy)."""
    env = FrameSwitchEnv(frames, switch_period=SWITCH_PERIOD, jitter=JITTER, noise_sigma=NOISE_SIGMA, seed=seed)
    agent = LearnedEmbeddingAgent(len(frames), seed=seed, initial_experts=initial_experts)
    rewards = []
    for _ in range(N_STEPS_B):
        info = env.observe()
        reward = env.feedback(agent.act(info.raw))
        agent.update(reward)
        rewards.append(reward)
    early = sum(rewards[:EARLY_WINDOW]) / EARLY_WINDOW
    overall = sum(rewards) / len(rewards)
    return early, overall


CONDITIONS = ("isomorphic", "non_isomorphic")
STARTS = ("transfer", "random_differentiated", "scratch")


def main() -> None:
    results: dict[str, dict[str, list[float]]] = {
        cond: {f"{start}_{metric}": [] for start in STARTS for metric in ("early", "overall")}
        for cond in CONDITIONS
    }

    for trial in TRIALS:
        trained_experts = train_on_domain_a(seed=trial)
        random_experts = random_differentiated_experts(len(DOMAIN_A_FRAMES), seed=trial)
        b_seed = trial + 1000
        for cond, frames in (("isomorphic", DOMAIN_B_ISOMORPHIC_FRAMES), ("non_isomorphic", DOMAIN_B_NONISOMORPHIC_FRAMES)):
            for start, experts in (("transfer", trained_experts), ("random_differentiated", random_experts), ("scratch", None)):
                early, overall = run_on_domain_b(frames, b_seed, initial_experts=experts)
                results[cond][f"{start}_early"].append(early)
                results[cond][f"{start}_overall"].append(overall)

    print(f"{'condition':<16} {'transfer':>10} {'rand_diff':>10} {'scratch':>10}   (early window)")
    for cond in CONDITIONS:
        r = results[cond]
        print(f"{cond:<16} {statistics.mean(r['transfer_early']):>10.3f} {statistics.mean(r['random_differentiated_early']):>10.3f} "
              f"{statistics.mean(r['scratch_early']):>10.3f}")
    print(f"{'condition':<16} {'transfer':>10} {'rand_diff':>10} {'scratch':>10}   (overall)")
    for cond in CONDITIONS:
        r = results[cond]
        print(f"{cond:<16} {statistics.mean(r['transfer_overall']):>10.3f} {statistics.mean(r['random_differentiated_overall']):>10.3f} "
              f"{statistics.mean(r['scratch_overall']):>10.3f}")

    print()
    for cond in CONDITIONS:
        r = results[cond]
        for metric in ("early", "overall"):
            # transfer vs scratch: total apparent advantage
            total_diffs = [t - s for t, s in zip(r[f"transfer_{metric}"], r[f"scratch_{metric}"])]
            # transfer vs random_differentiated: advantage attributable to *informed* values specifically,
            # isolating it from "any pre-differentiated start helps"
            structure_diffs = [t - rd for t, rd in zip(r[f"transfer_{metric}"], r[f"random_differentiated_{metric}"])]
            # random_differentiated vs scratch: the confound alone
            confound_diffs = [rd - s for rd, s in zip(r[f"random_differentiated_{metric}"], r[f"scratch_{metric}"])]
            for label, diffs in (("transfer-scratch (total)", total_diffs),
                                  ("transfer-rand_diff (structure-specific)", structure_diffs),
                                  ("rand_diff-scratch (confound)", confound_diffs)):
                wins = sum(1 for d in diffs if d > 0)
                print(f"{cond:<16} {metric:<8} {label:<42} mean={statistics.mean(diffs):+.3f} "
                      f"stdev={statistics.pstdev(diffs):.3f} wins={wins}/{len(diffs)}")


if __name__ == "__main__":
    main()
