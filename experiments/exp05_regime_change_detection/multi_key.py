"""Experiment 5, multi-key tracking (docs/research-agenda.md #7b) -- does
combining evidence across keys BEFORE thresholding (joint_change_points,
Crosier 1988) improve detection over running keys independently, or is any
apparent improvement just "two independent chances to detect" (a union of
per-key detections)? The second is the confound to control for -- the same
kind of control experiments 3 and 4 needed before trusting a positive
result (docs/related-work.md §9a; docs/research-agenda.md #7a).

Two keys share the SAME true regime-change point and the SAME shift
direction/magnitude, but with independent noise -- a scenario deliberately
chosen so each key alone is a weak (~44% recall) detector, to give a
combination method room to show a real gain if it has one.

Run: PYTHONPATH=. python experiments/exp05_regime_change_detection/multi_key.py
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone

from transintelligence import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner

from experiments.exp05_regime_change_detection.run import TOLERANCE, evaluate

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOISE_SIGMA = 0.05
REGIME_LENGTH = 60
SEEDS = list(range(50))
KEYS = ["a", "b"]


def make_history(seed: int, shift: float) -> tuple[StateHistory, int]:
    rng = random.Random(seed)
    states = []
    step = 0
    for mu in (0.5, 0.5 + shift):
        for _ in range(REGIME_LENGTH):
            vals = {k: mu + rng.gauss(0, NOISE_SIGMA) for k in KEYS}
            states.append(State("x", vals, T0 + timedelta(minutes=step)))
            step += 1
    return StateHistory(states), REGIME_LENGTH


def detected_steps(changes: list[datetime]) -> list[int]:
    return [round((c - T0).total_seconds() / 60) for c in changes]


def run_trial(seed: int, shift: float) -> dict:
    history, true_step = make_history(seed, shift)
    true_change_steps = [true_step]
    reasoner = CUSUMTemporalReasoner()

    per_key_steps = {k: detected_steps(reasoner.change_points(history, k)) for k in KEYS}
    union_steps = sorted(set(per_key_steps["a"]) | set(per_key_steps["b"]))
    joint_steps = detected_steps(reasoner.joint_change_points(history, KEYS))

    return {
        "recall_a": evaluate(per_key_steps["a"], true_change_steps)["recall"],
        "recall_b": evaluate(per_key_steps["b"], true_change_steps)["recall"],
        "recall_union": evaluate(union_steps, true_change_steps)["recall"],
        "recall_joint": evaluate(joint_steps, true_change_steps)["recall"],
        "delay_union": evaluate(union_steps, true_change_steps)["mean_delay"],
        "delay_joint": evaluate(joint_steps, true_change_steps)["mean_delay"],
    }


def main() -> None:
    for shift in (0.02, 0.03, 0.04, 0.05, 0.06):
        rows = [run_trial(seed, shift) for seed in SEEDS]
        recall_a = statistics.mean(r["recall_a"] for r in rows)
        recall_b = statistics.mean(r["recall_b"] for r in rows)
        recall_union = statistics.mean(r["recall_union"] for r in rows)
        recall_joint = statistics.mean(r["recall_joint"] for r in rows)
        delays_union = [r["delay_union"] for r in rows if r["delay_union"] == r["delay_union"]]
        delays_joint = [r["delay_joint"] for r in rows if r["delay_joint"] == r["delay_joint"]]

        # Paired win/tie/loss on the same seeds -- more robust than the
        # raw mean-recall gap at these sample sizes (n=50).
        joint_wins = sum(1 for r in rows if r["recall_joint"] > r["recall_union"])
        union_wins = sum(1 for r in rows if r["recall_union"] > r["recall_joint"])
        ties = len(rows) - joint_wins - union_wins

        print(f"shift={shift} (shift/noise={shift/NOISE_SIGMA:.2f}): "
              f"recall_a={recall_a:.2f} recall_b={recall_b:.2f} "
              f"recall_union={recall_union:.2f} recall_joint={recall_joint:.2f} "
              f"delay_union={statistics.mean(delays_union) if delays_union else float('nan'):.2f} "
              f"delay_joint={statistics.mean(delays_joint) if delays_joint else float('nan'):.2f} "
              f"paired[joint_wins={joint_wins} union_wins={union_wins} ties={ties}]")


if __name__ == "__main__":
    main()
