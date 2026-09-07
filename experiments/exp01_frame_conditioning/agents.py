"""Agent conditions for Experiment 1 (docs/research-agenda.md #5).

All three agents face the same partially-observable problem: predict
sign(evaluate(raw, active_frame)) for a raw value, without being told which
of the known candidate frames is currently active, then learn from a
binary (correct/incorrect) reward. `FlatOracleAgent` is the exception --
it is deliberately given the true active frame index, as the control the
falsification criterion calls for.
"""
from __future__ import annotations

from transintelligence import Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner


class RFAwareAgent:
    """Maintains a belief distribution over the known candidate frames and
    predicts via a belief-weighted vote of evaluate(x, frame) across them --
    a small Bayesian forward filter (HMM-style) over which frame is active,
    using the real BaselineRelativeReasoner/evaluate() rather than a
    reimplementation of the reasoning logic."""

    name = "rf_aware"

    def __init__(self, frames: list[ReferenceFrame], switch_prob: float = 1 / 40, error_rate: float = 0.05):
        self.frames = frames
        self.belief = [1.0 / len(frames)] * len(frames)
        self.switch_prob = switch_prob
        self.error_rate = error_rate
        self._entity = Entity("synthetic", "x")
        self._last_conclusions: list[int] = []
        self._last_prediction = 1

    def act(self, raw: float) -> int:
        reasoner = BaselineRelativeReasoner([Observation(self._entity.id, "score", raw, "agent")])
        self._last_conclusions = [1 if reasoner.evaluate(self._entity, f).result > 0 else -1 for f in self.frames]
        vote = sum(b * c for b, c in zip(self.belief, self._last_conclusions))
        self._last_prediction = 1 if vote >= 0 else -1
        return self._last_prediction

    def update(self, reward: int) -> None:
        eps = self.error_rate
        likelihoods = []
        for c in self._last_conclusions:
            matches = c == self._last_prediction
            p_reward_1 = (1 - eps) if matches else eps
            likelihoods.append(p_reward_1 if reward == 1 else (1 - p_reward_1))
        posterior = [b * l for b, l in zip(self.belief, likelihoods)]
        total = sum(posterior) or 1.0
        posterior = [p / total for p in posterior]
        uniform = 1.0 / len(self.frames)
        self.belief = [(1 - self.switch_prob) * p + self.switch_prob * uniform for p in posterior]

    def belief_in(self, frame_index: int) -> float:
        return self.belief[frame_index]


class FlatBaselineAgent:
    """No ReferenceFrame structure: a single online-adapting linear rule
    (perceptron-style) over the raw scalar. Must re-adapt the same two
    parameters from scratch whenever the active rule changes -- no factored
    representation of "which of several known rules is active"."""

    name = "flat"

    def __init__(self, lr: float = 0.2):
        self.w = 0.0
        self.b = 0.0
        self.lr = lr
        self._last_raw = 0.0
        self._last_prediction = 1

    def act(self, raw: float) -> int:
        self._last_raw = raw
        score = self.w * raw + self.b
        self._last_prediction = 1 if score >= 0 else -1
        return self._last_prediction

    def update(self, reward: int) -> None:
        if reward == 0:
            target = -self._last_prediction
            self.w += self.lr * target * self._last_raw
            self.b += self.lr * target


class FlatOracleAgent:
    """Control condition: same flat linear-rule mechanism as
    FlatBaselineAgent, but given the true active frame index directly and
    kept as N independent per-frame-id rules -- the deliberately unfair
    comparison the falsification criterion calls for. If RFAwareAgent's
    advantage over FlatBaselineAgent vanishes against this, the advantage
    was information access, not structure."""

    name = "flat_oracle"

    def __init__(self, n_frames: int, lr: float = 0.2):
        self._rules = [FlatBaselineAgent(lr) for _ in range(n_frames)]
        self._active_rule: FlatBaselineAgent | None = None

    def act(self, raw: float, frame_index: int) -> int:
        self._active_rule = self._rules[frame_index]
        return self._active_rule.act(raw)

    def update(self, reward: int) -> None:
        assert self._active_rule is not None, "call act() first"
        self._active_rule.update(reward)
