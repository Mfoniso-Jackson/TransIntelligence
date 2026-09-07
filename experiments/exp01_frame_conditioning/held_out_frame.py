"""Held-out-frame test (the last item on RESULTS.md's "what this still does
not establish" list): RFAwareAgent's candidate frame list is fixed at
construction. What happens when the environment's active frame is genuinely
NOT in that list?

This is a real limitation worth testing honestly, not just flagging: an
agent with a fixed, explicit, finite set of candidate hypotheses cannot
represent a truly novel regime by construction, whereas FlatBaselineAgent
and LearnedEmbeddingAgent never assumed a known frame set to begin with --
they might cope with a genuinely novel threshold rule just fine, since
their mechanism doesn't require having enumerated it in advance. This
experiment measures whether that asymmetry actually shows up, and how
badly, rather than assuming the answer either way.

Run: PYTHONPATH=. python experiments/exp01_frame_conditioning/held_out_frame.py
"""
from __future__ import annotations

import statistics

from transintelligence import ReferenceFrame

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.agents import FlatBaselineAgent, LearnedEmbeddingAgent, RFAwareAgent

ALL_FRAMES = [
    ReferenceFrame("f1", baseline=0.5, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f2", baseline=0.3, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f3", baseline=0.5, metadata={"direction": "lower_is_better"}),
    ReferenceFrame("f4", baseline=0.7, metadata={"direction": "lower_is_better"}),  # held out from rf_aware
]
TRAIN_FRAMES = ALL_FRAMES[:3]
HELD_OUT_INDEX = 3

N_STEPS = 3000
SWITCH_PERIOD = 40
JITTER = 10
NOISE_SIGMA = 0.05
SEEDS = list(range(10))


def run_seed(agent_kind: str, seed: int) -> tuple[float, float]:
    """Returns (accuracy while a known frame is active, accuracy while the held-out frame is active)."""
    env = FrameSwitchEnv(ALL_FRAMES, switch_period=SWITCH_PERIOD, jitter=JITTER, noise_sigma=NOISE_SIGMA, seed=seed)
    if agent_kind == "rf_aware":
        agent = RFAwareAgent(TRAIN_FRAMES)  # does NOT know about f4
    elif agent_kind == "flat":
        agent = FlatBaselineAgent()
    elif agent_kind == "learned_embedding":
        agent = LearnedEmbeddingAgent(len(TRAIN_FRAMES), seed=seed)
    else:
        raise ValueError(agent_kind)

    known_correct = known_total = 0
    held_out_correct = held_out_total = 0
    for _ in range(N_STEPS):
        info = env.observe()
        predict = agent.act(info.raw)
        reward = env.feedback(predict)
        agent.update(reward)
        if info.active_frame_index == HELD_OUT_INDEX:
            held_out_correct += reward
            held_out_total += 1
        else:
            known_correct += reward
            known_total += 1
    return known_correct / known_total, held_out_correct / held_out_total


def main() -> None:
    print(f"{'agent':<18} {'acc_known_frames':>17} {'acc_held_out_frame':>19} {'gap':>8}")
    for agent_kind in ("flat", "learned_embedding", "rf_aware"):
        known_accs, held_out_accs = [], []
        for seed in SEEDS:
            known, held_out = run_seed(agent_kind, seed)
            known_accs.append(known)
            held_out_accs.append(held_out)
        known_mean = statistics.mean(known_accs)
        held_out_mean = statistics.mean(held_out_accs)
        print(f"{agent_kind:<18} {known_mean:>17.3f} {held_out_mean:>19.3f} {held_out_mean - known_mean:>+8.3f}")


if __name__ == "__main__":
    main()
