"""Experiment 2 — does sensitivity() detect frame-dependent conclusions?

See docs/research-agenda.md #6 and docs/related-work.md #4 for the
hypothesis and why this experiment is cheap enough to run first.

Ground truth: for a single entity with noiseless true value ``mu``, a
frame R's "conclusion" is ``sign(evaluate(x, R).result)`` — whether the
entity looks good (positive) or bad (negative) relative to that frame's
baseline/direction. An (entity, frame-pair) instance is frame-dependent
(label=1) if the noiseless conclusion flips between the two frames,
frame-invariant (label=0) otherwise.

We test whether ``BaselineRelativeReasoner.sensitivity()`` — the only
detector the codebase currently provides — predicts this label, using
real `Entity`/`Observation`/`ReferenceFrame`/`BaselineRelativeReasoner`
objects (no reimplementation of the reasoner), at varying observation
noise, for two regimes: frame pairs that share a ``direction`` and frame
pairs with opposite ``direction``.

Run: python experiments/exp02_frame_dependence/run.py
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from transintelligence import Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner

PROPERTY = "score"
N_ENTITIES = 300
NOISE_SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
SEED = 0


def make_frame(name: str, baseline: float, direction: str) -> ReferenceFrame:
    return ReferenceFrame(name, baseline=baseline, metadata={"property": PROPERTY, "direction": direction})


def true_conclusion(mu: float, frame: ReferenceFrame) -> int:
    """Noiseless sign(evaluate(x, frame)) computed from the ground-truth mu, independent of the reasoner."""
    score = mu - frame.baseline
    if frame.metadata["direction"] == "lower_is_better":
        score = -score
    return 1 if score > 0 else -1


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (P(score_pos > score_neg), ties count as 0.5). O(n^2), fine at this scale."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


@dataclass
class ConditionResult:
    condition: str
    sigma: float
    auc_sensitivity: float
    noisy_conclusion_accuracy: float
    label_positive_rate: float
    sensitivity_stdev: float


def run_condition(condition: str, r1: ReferenceFrame, r2: ReferenceFrame, sigma: float, rng: random.Random) -> ConditionResult:
    sens_scores: list[float] = []
    labels: list[int] = []
    noisy_correct = 0

    for i in range(N_ENTITIES):
        mu = rng.uniform(0.0, 1.0)
        label = 1 if true_conclusion(mu, r1) != true_conclusion(mu, r2) else 0
        labels.append(label)

        observed = mu + rng.gauss(0.0, sigma)
        entity = Entity("synthetic", f"e{i}")
        reasoner = BaselineRelativeReasoner([Observation(entity.id, PROPERTY, observed, "synthetic", 0.9)])

        # sensitivity() is negative when the conclusion flips (frame-dependent)
        # and positive when it agrees (frame-invariant) -- negate so higher
        # score means "more likely frame-dependent", matching label=1.
        sens_scores.append(-reasoner.sensitivity(entity, r1, r2).result)

        noisy_label = 1 if (reasoner.evaluate(entity, r1).result > 0) != (reasoner.evaluate(entity, r2).result > 0) else 0
        noisy_correct += int(noisy_label == label)

    return ConditionResult(
        condition=condition,
        sigma=sigma,
        auc_sensitivity=auc(sens_scores, labels),
        noisy_conclusion_accuracy=noisy_correct / N_ENTITIES,
        label_positive_rate=sum(labels) / N_ENTITIES,
        sensitivity_stdev=statistics.pstdev(sens_scores),
    )


def main() -> None:
    rng = random.Random(SEED)
    r1_same = make_frame("historical", baseline=0.5, direction="higher_is_better")
    r2_same = make_frame("portfolio-risk", baseline=0.3, direction="higher_is_better")
    r1_diff = make_frame("historical", baseline=0.5, direction="higher_is_better")
    r2_diff = make_frame("portfolio-risk", baseline=0.3, direction="lower_is_better")

    print(f"{'condition':<14} {'sigma':>6} {'AUC(sensitivity)':>18} {'noisy_concl_acc':>16} {'pos_rate':>9} {'sens_stdev':>11}")
    for sigma in NOISE_SIGMAS:
        for condition, r1, r2 in (("same_dir", r1_same, r2_same), ("diff_dir", r1_diff, r2_diff)):
            res = run_condition(condition, r1, r2, sigma, rng)
            print(f"{res.condition:<14} {res.sigma:>6.2f} {res.auc_sensitivity:>18.3f} "
                  f"{res.noisy_conclusion_accuracy:>16.3f} {res.label_positive_rate:>9.3f} {res.sensitivity_stdev:>11.4f}")


if __name__ == "__main__":
    main()
