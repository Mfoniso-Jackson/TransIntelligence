"""Experiment 4 -- frame discovery (docs/research-agenda.md #7a).

Reuses experiment 1's held-out-frame setup exactly (ALL_FRAMES,
TRAIN_FRAMES, environment config) so this experiment's baseline numbers
are directly comparable to experiments/exp01_frame_conditioning/RESULTS.md.

Three agents, same trigger mechanism, different response to it:
  - static (RFAwareAgent, no discovery): the experiment 1 baseline.
  - discovering_rf: fits a new frame from the triggering window's data.
  - random_discovery: appends a random frame on the identical trigger --
    the confound control (docs/research-agenda.md #7a).

Run: PYTHONPATH=. python experiments/exp04_frame_discovery/run.py
"""
from __future__ import annotations

import statistics

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.agents import RFAwareAgent
from experiments.exp01_frame_conditioning.held_out_frame import ALL_FRAMES, HELD_OUT_INDEX, TRAIN_FRAMES
from experiments.exp04_frame_discovery.agents import DiscoveringRFAgent, RandomDiscoveryAgent

N_STEPS = 3000
SWITCH_PERIOD = 40
JITTER = 10
NOISE_SIGMA = 0.05
SEEDS = list(range(10))
WINDOW = 40
MIN_ACCURACY = 0.85
ALPHA = 0.01


def run_seed(agent_kind: str, seed: int) -> dict:
    env = FrameSwitchEnv(ALL_FRAMES, switch_period=SWITCH_PERIOD, jitter=JITTER, noise_sigma=NOISE_SIGMA, seed=seed)
    if agent_kind == "static":
        agent = RFAwareAgent(TRAIN_FRAMES)
    elif agent_kind == "discovering_rf":
        agent = DiscoveringRFAgent(TRAIN_FRAMES, window=WINDOW, min_accuracy=MIN_ACCURACY, alpha=ALPHA, seed=seed)
    elif agent_kind == "random_discovery":
        agent = RandomDiscoveryAgent(TRAIN_FRAMES, window=WINDOW, min_accuracy=MIN_ACCURACY, alpha=ALPHA, seed=seed)
    else:
        raise ValueError(agent_kind)

    held_out_rewards_before: list[int] = []
    held_out_rewards_after: list[int] = []
    first_discovery_step = None

    for step in range(N_STEPS):
        info = env.observe()
        is_held_out = info.active_frame_index == HELD_OUT_INDEX
        predict = agent.act(info.raw)
        reward = env.feedback(predict)
        if agent_kind == "static":
            agent.update(reward)
        else:
            agent.update(reward, true_is_held_out=is_held_out)
            if first_discovery_step is None:
                first_discovery_step = agent.first_acted_discovery_step

        if is_held_out:
            if first_discovery_step is not None and step >= first_discovery_step:
                held_out_rewards_after.append(reward)
            else:
                held_out_rewards_before.append(reward)

    result = {
        "acc_held_out_before": sum(held_out_rewards_before) / len(held_out_rewards_before) if held_out_rewards_before else float("nan"),
        "acc_held_out_after": sum(held_out_rewards_after) / len(held_out_rewards_after) if held_out_rewards_after else float("nan"),
        "first_discovery_step": first_discovery_step,
    }
    if agent_kind != "static":
        trigger_log = agent.trigger_log
        false_triggers = [t for t in trigger_log if t["held_out_fraction"] < 0.5]
        result["n_triggers"] = len(trigger_log)
        result["n_false_triggers"] = len(false_triggers)
        result["false_discovery_rate"] = len(false_triggers) / len(trigger_log) if trigger_log else float("nan")
        result["final_frame_count"] = agent.frame_count
    return result


def main() -> None:
    for agent_kind in ("static", "discovering_rf", "random_discovery"):
        rows = [run_seed(agent_kind, seed) for seed in SEEDS]
        before = [r["acc_held_out_before"] for r in rows if r["acc_held_out_before"] == r["acc_held_out_before"]]  # filter nan
        after = [r["acc_held_out_after"] for r in rows if r["acc_held_out_after"] == r["acc_held_out_after"]]
        discovered = [r for r in rows if r.get("first_discovery_step") is not None]

        print(f"--- {agent_kind} ---")
        print(f"  held-out accuracy BEFORE first discovery: mean={statistics.mean(before):.3f} (n_seeds_with_data={len(before)})")
        if after:
            print(f"  held-out accuracy AFTER  first discovery: mean={statistics.mean(after):.3f} (n_seeds_with_data={len(after)})")
        else:
            print(f"  held-out accuracy AFTER  first discovery: no data (no seed discovered)")
        print(f"  seeds that discovered at least once: {len(discovered)}/{len(rows)}")
        if agent_kind != "static":
            fdrs = [r["false_discovery_rate"] for r in rows if r["false_discovery_rate"] == r["false_discovery_rate"]]
            frame_counts = [r["final_frame_count"] for r in rows]
            n_triggers_total = sum(r["n_triggers"] for r in rows)
            print(f"  mean false-discovery rate (of windows that triggered): {statistics.mean(fdrs):.3f} (total triggers across seeds: {n_triggers_total})")
            print(f"  mean final frame count: {statistics.mean(frame_counts):.2f} (started at {len(TRAIN_FRAMES)})")
        print()


if __name__ == "__main__":
    main()
