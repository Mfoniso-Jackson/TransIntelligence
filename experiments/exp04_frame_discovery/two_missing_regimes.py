"""Experiment 4, extended to two simultaneously-missing regimes
(docs/research-agenda.md #7a "what this does not establish").

5 total frames, 3 known, 2 held out (chosen to be mutually distinct --
different baselines AND different directions from each other, not just
from the known set, so a fit for one shouldn't accidentally help with the
other). Tracks each held-out regime's own accuracy, split by whether a
frame close to *that specific* regime has been discovered yet -- discovering
regime A's frame shouldn't be credited for regime B's recovery.

Run: PYTHONPATH=. python experiments/exp04_frame_discovery/two_missing_regimes.py
"""
from __future__ import annotations

import statistics

from transintelligence import ReferenceFrame

from environments.transworld import FrameSwitchEnv
from experiments.exp04_frame_discovery.agents import DiscoveringRFAgent

ALL_FRAMES = [
    ReferenceFrame("f1", baseline=0.5, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f2", baseline=0.3, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f3", baseline=0.5, metadata={"direction": "lower_is_better"}),
    ReferenceFrame("f4", baseline=0.7, metadata={"direction": "lower_is_better"}),  # held out
    ReferenceFrame("f5", baseline=0.2, metadata={"direction": "higher_is_better"}),  # held out, distinct from f4
]
TRAIN_FRAMES = ALL_FRAMES[:3]
HELD_OUT_INDICES = [3, 4]
MAX_FRAMES = len(TRAIN_FRAMES) + 4  # generous slack for two discoveries plus the occasional retry

SWITCH_PERIOD = 40
JITTER = 10
NOISE_SIGMA = 0.05
N_STEPS = 4500  # longer than the single-held-out run: two regimes to find, not one
SEEDS = list(range(10))
WINDOW = 40
MIN_ACCURACY = 0.85
ALPHA = 0.01
MATCH_TOLERANCE = 0.15


def _matches(frame: ReferenceFrame, true_frame: ReferenceFrame) -> bool:
    return (abs(frame.baseline - true_frame.baseline) < MATCH_TOLERANCE
            and frame.metadata["direction"] == true_frame.metadata["direction"])


def run_seed(seed: int) -> dict:
    env = FrameSwitchEnv(ALL_FRAMES, switch_period=SWITCH_PERIOD, jitter=JITTER, noise_sigma=NOISE_SIGMA, seed=seed)
    agent = DiscoveringRFAgent(TRAIN_FRAMES, window=WINDOW, min_accuracy=MIN_ACCURACY, alpha=ALPHA,
                                max_frames=MAX_FRAMES, seed=seed)

    per_frame_rewards: dict[int, list[tuple[int, int]]] = {idx: [] for idx in HELD_OUT_INDICES}  # (step, reward)
    for step in range(N_STEPS):
        info = env.observe()
        is_held_out = info.active_frame_index in HELD_OUT_INDICES
        reward = env.feedback(agent.act(info.raw))
        agent.update(reward, true_is_held_out=is_held_out)
        if info.active_frame_index in HELD_OUT_INDICES:
            per_frame_rewards[info.active_frame_index].append((step, reward))

    # For each held-out frame, find the earliest step a matching frame was discovered.
    first_match_step: dict[int, int | None] = {idx: None for idx in HELD_OUT_INDICES}
    for entry in agent.trigger_log:
        if not entry["acted"] or entry["frame"] is None:
            continue
        for idx in HELD_OUT_INDICES:
            if first_match_step[idx] is None and _matches(entry["frame"], ALL_FRAMES[idx]):
                first_match_step[idx] = entry["step"]

    result = {}
    for idx in HELD_OUT_INDICES:
        rewards = per_frame_rewards[idx]
        match_step = first_match_step[idx]
        before = [r for s, r in rewards if match_step is None or s < match_step]
        after = [r for s, r in rewards if match_step is not None and s >= match_step]
        result[idx] = {
            "discovered": match_step is not None,
            "acc_before": sum(before) / len(before) if before else float("nan"),
            "acc_after": sum(after) / len(after) if after else float("nan"),
        }
    result["final_frame_count"] = agent.frame_count
    result["n_triggers"] = len(agent.trigger_log)
    result["n_false_triggers"] = sum(1 for t in agent.trigger_log if t["held_out_fraction"] < 0.5)
    return result


def main() -> None:
    rows = [run_seed(seed) for seed in SEEDS]
    for idx in HELD_OUT_INDICES:
        disc_rate = sum(r[idx]["discovered"] for r in rows) / len(rows)
        before_vals = [r[idx]["acc_before"] for r in rows if r[idx]["acc_before"] == r[idx]["acc_before"]]
        after_vals = [r[idx]["acc_after"] for r in rows if r[idx]["acc_after"] == r[idx]["acc_after"]]
        print(f"held-out frame index {idx} ({ALL_FRAMES[idx].baseline}, {ALL_FRAMES[idx].metadata['direction']}):")
        print(f"  discovered in {sum(r[idx]['discovered'] for r in rows)}/{len(rows)} seeds")
        print(f"  acc before match: {statistics.mean(before_vals):.3f}" if before_vals else "  acc before match: n/a")
        print(f"  acc after match:  {statistics.mean(after_vals):.3f}" if after_vals else "  acc after match:  n/a")

    both_discovered = sum(1 for r in rows if all(r[idx]["discovered"] for idx in HELD_OUT_INDICES))
    print(f"\nboth regimes discovered in the same seed: {both_discovered}/{len(rows)}")
    print(f"mean final frame count: {statistics.mean(r['final_frame_count'] for r in rows):.2f} (started at {len(TRAIN_FRAMES)}, cap {MAX_FRAMES})")
    total_triggers = sum(r["n_triggers"] for r in rows)
    total_false = sum(r["n_false_triggers"] for r in rows)
    print(f"false-discovery rate: {total_false}/{total_triggers} = {total_false/total_triggers if total_triggers else float('nan'):.3f}")


if __name__ == "__main__":
    main()
