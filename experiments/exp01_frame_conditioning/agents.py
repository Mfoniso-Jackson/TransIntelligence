"""Agent conditions for Experiment 1 (docs/research-agenda.md #5).

Most of these agents face the same partially-observable problem: predict
sign(evaluate(raw, active_frame)) for a raw value, without being told which
of the known candidate frames is currently active, then learn from a
binary (correct/incorrect) reward. `FlatOracleAgent` and `TrueOracleAgent`
are the exceptions -- both are given the true active frame index directly,
as controls.

`LearnedEmbeddingAgent` is condition B (docs/related-work.md #2): the same
multi-hypothesis Bayesian belief-tracking *architecture* as `RFAwareAgent`
(same number of slots, same belief-update mechanic), but each slot is an
opaque linear rule learned online from reward feedback rather than a known
`ReferenceFrame` with human-legible baseline/direction fields. This is the
comparison that actually tests explicit/legible structure against a
capacity-matched implicit one, rather than against a flat single-rule
baseline.
"""
from __future__ import annotations

import math
import random

from transintelligence import Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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


class TrueOracleAgent:
    """Ceiling control: given the true active frame index directly *and*
    allowed to call the real evaluate() on the correct known ReferenceFrame
    -- no inference and no learning needed, the rule is exactly known.
    Unlike FlatOracleAgent (which knows *which* frame is active but still
    has to learn *what its rule is*), this isolates "cost of inference"
    as the entire gap between it and RFAwareAgent -- see RESULTS.md for why
    FlatOracleAgent alone conflates the two."""

    name = "true_oracle"

    def __init__(self, frames: list[ReferenceFrame]):
        self.frames = frames
        self._entity = Entity("synthetic", "x")

    def act(self, raw: float, frame_index: int) -> int:
        reasoner = BaselineRelativeReasoner([Observation(self._entity.id, "score", raw, "agent")])
        return 1 if reasoner.evaluate(self._entity, self.frames[frame_index]).result > 0 else -1

    def update(self, reward: int) -> None:
        pass  # nothing to learn -- the rule is already exactly known


class LearnedEmbeddingAgent:
    """Condition B: same multi-hypothesis Bayesian belief-tracking mechanism
    as RFAwareAgent (n_slots slots, identical belief-update math), but each
    slot is an opaque linear rule (w, b) learned online, rather than a known
    ReferenceFrame. No ReferenceFrame object, no evaluate() call -- the
    slots must discover which raw-value thresholds matter from reward
    feedback alone, same as FlatBaselineAgent, but with n_slots independent
    hypotheses tracked instead of one.

    Weight updates use hard (argmax) responsibility assignment: only the
    single currently-most-likely slot is updated per step, at full learning
    rate, rather than spreading a belief-weighted fraction of the update
    across all slots. An earlier soft-weighted version underperformed the
    flat baseline (see RESULTS.md) -- diluting the update across all
    n_slots slots every step is slower to specialize than a single rule
    that commits fully to whatever it's currently seeing. Belief itself
    (used for the *prediction* vote and for choosing which slot to update)
    stays the same soft Bayesian filter as RFAwareAgent -- only the
    weight-update assignment is hardened.

    The winning slot is updated via an online logistic-regression gradient
    step, not a perceptron mistake-rule: reward + this agent's own last
    prediction together imply the true label on *every* step (if reward=1
    the label was the prediction, if reward=0 it was the opposite), so a
    real supervised gradient step is available every step, not just on
    mistakes -- a materially closer match to how prior art (arXiv:2102.06177,
    arXiv:2207.02249, see docs/related-work.md #2) actually trains context
    representations, rather than a binary correct/incorrect nudge.

    A fourth version tried adding a shared exponential moving average of
    recent reward as a second input feature per slot, to give the agent
    some *temporal* context beyond the current raw value (closer to how
    the cited papers infer context from a reward trajectory). It made no
    measurable difference (RESULTS.md) -- the belief vector here already
    carries the relevant temporal signal, so a redundant per-slot feature
    added parameters without benefit. Reverted; noted for anyone tempted
    to retry the same idea."""

    name = "learned_embedding"

    def __init__(self, n_slots: int, lr: float = 0.2, switch_prob: float = 1 / 40, error_rate: float = 0.05,
                 init_scale: float = 0.05, seed: int = 0):
        # Symmetry-breaking is load-bearing here: slots initialized at
        # exactly (0,0) receive identical proportionally-scaled updates
        # forever (sign(c*x) == sign(x) for any c>0), so belief never
        # differentiates and the whole ensemble degenerates to a slowed-down
        # copy of a single flat rule -- confirmed by an earlier run that
        # produced numerically identical results to FlatBaselineAgent (see
        # RESULTS.md). Small random initialization breaks that degeneracy.
        rng = random.Random(seed)
        self.n_slots = n_slots
        self.experts: list[tuple[float, float]] = [(rng.gauss(0, init_scale), rng.gauss(0, init_scale)) for _ in range(n_slots)]
        self.belief = [1.0 / n_slots] * n_slots
        self.lr = lr
        self.switch_prob = switch_prob
        self.error_rate = error_rate
        self._last_raw = 0.0
        self._last_slot_predictions: list[int] = []
        self._last_prediction = 1
        self._last_winner = 0

    def act(self, raw: float) -> int:
        self._last_raw = raw
        self._last_slot_predictions = [1 if (w * raw + b) >= 0 else -1 for w, b in self.experts]
        self._last_winner = max(range(self.n_slots), key=lambda i: self.belief[i])
        vote = sum(b_i * p for b_i, p in zip(self.belief, self._last_slot_predictions))
        self._last_prediction = 1 if vote >= 0 else -1
        return self._last_prediction

    def update(self, reward: int) -> None:
        eps = self.error_rate
        likelihoods = []
        for p in self._last_slot_predictions:
            matches = p == self._last_prediction
            p_reward_1 = (1 - eps) if matches else eps
            likelihoods.append(p_reward_1 if reward == 1 else (1 - p_reward_1))
        posterior = [b * l for b, l in zip(self.belief, likelihoods)]
        total = sum(posterior) or 1.0
        posterior = [p / total for p in posterior]
        uniform = 1.0 / self.n_slots
        self.belief = [(1 - self.switch_prob) * p + self.switch_prob * uniform for p in posterior]

        # Reward + our own last prediction imply the true label on every
        # step (not just mistakes): reward=1 -> label was the prediction,
        # reward=0 -> label was the opposite. Real logistic-regression
        # gradient step on the winning slot toward that implied label.
        implied_target01 = 1.0 if (self._last_prediction if reward == 1 else -self._last_prediction) > 0 else 0.0
        w, b = self.experts[self._last_winner]
        p_hat = _sigmoid(w * self._last_raw + b)
        grad = p_hat - implied_target01
        self.experts[self._last_winner] = (w - self.lr * grad * self._last_raw, b - self.lr * grad)


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
