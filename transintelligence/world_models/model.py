"""Baseline world models: learned forward dynamics for planning (Phase 6,
docs/research-agenda.md #7h, docs/master-context.md §13/§19).

`transintelligence/world_models/` was, before this, a directory that
didn't exist at all, and `Predictor` in `reasoning/interfaces.py`
declared no methods -- the same starting point every other empty-stub
reasoning protocol had before its phase started. The master context's
own formalization of a world model is `M(S_t, A_t) -> S_{t+1}`: not a
value estimate, a *transition* model, useful specifically because
predicted outcomes of different candidate actions from the same state can
then be compared (`S_t ->A_1-> S_{t+1}` vs `S_t ->A_2-> S'_{t+1}`) --
Sutton's Dyna architecture (*Integrated Architectures for Learning,
Planning, and Reacting Based on Approximating Dynamic Programming*, ICML
1990; *Dyna, an Integrated Architecture for Learning, Planning, and
Reacting*, SIGART Bulletin 1991) is the established mechanism this
implements the smallest version of: plan by simulating candidate actions
through a learned model of the environment, rather than only acting on
cached historical value. Ha & Schmidhuber's *World Models* (arXiv:1803.10122,
2018) is the modern, much heavier (generative/recurrent neural network)
version of the same "learn M, then act inside it" idea -- not the
mechanism implemented here, cited only as the paper that popularized the
term "world model" for this class of methods.

`LinearDynamicsModel` learns a per-action linear map `next_state =
intercept + slope * state`, fit via `ordinary_least_squares`
(`reasoning/causal/model.py`) from observed `(state, action, next_state)`
transitions -- the same linearity simplification `BaselineRelativeReasoner`,
`CUSUMTemporalReasoner`, and `reasoning/causal/`'s effect estimation
already make elsewhere in this codebase. This is deliberately a 1-step
model used for 1-step lookahead planning, not a multi-step rollout/
simulation engine -- the smallest mechanism that could produce a
falsifiable claim about whether explicit dynamics modeling helps at all,
not a claim to handle long-horizon planning (`Simulator`, `Planner`
remain empty stubs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

from transintelligence.reasoning.causal import ordinary_least_squares


@dataclass(frozen=True)
class LinearDynamicsModel:
    """A learned per-action linear transition model: for each action seen
    so far, `next_state = intercept[action] + slope[action] * state`,
    fit independently per action via OLS. Actions with fewer than 2
    observed transitions have no fitted coefficients and `predict()`
    raises for them -- there's no meaningful residual/intercept-vs-slope
    fit from a single point, so this is a caller error (not enough
    exploration yet), not something to silently paper over with a
    default guess."""

    coefficients: dict[Hashable, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, transitions: list[tuple[float, Hashable, float]]) -> "LinearDynamicsModel":
        """`transitions` is a list of (state, action, next_state) triples,
        typically all transitions observed so far -- refit from scratch
        each call, matching this repo's existing `ordinary_least_squares`
        (non-incremental) and the "simplicity over efficiency" stance
        already taken for `CausalGraph`'s path enumeration."""
        by_action: dict[Hashable, list[tuple[float, float]]] = {}
        for state, action, next_state in transitions:
            by_action.setdefault(action, []).append((state, next_state))

        coefficients: dict[Hashable, tuple[float, float]] = {}
        for action, pairs in by_action.items():
            if len(pairs) < 2:
                continue
            features = [[1.0, s] for s, _ in pairs]
            targets = [ns for _, ns in pairs]
            try:
                intercept, slope = ordinary_least_squares(features, targets)
            except ValueError:
                continue  # degenerate (e.g. identical states seen for this action) -- skip, not enough signal yet
            coefficients[action] = (intercept, slope)
        return cls(coefficients=coefficients)

    def known_actions(self) -> set[Hashable]:
        return set(self.coefficients.keys())

    def predict(self, state: float, action: Hashable) -> float:
        if action not in self.coefficients:
            raise ValueError(f"no fitted dynamics for action {action!r} yet -- not enough transitions observed")
        intercept, slope = self.coefficients[action]
        return intercept + slope * state
