"""Baseline causal reasoning: causal graphs, d-separation, the backdoor
criterion, and linear-adjustment effect estimation (Phase 5,
docs/research-agenda.md #7c).

`transintelligence/reasoning/causal/` was, before this, a single-line
docstring stub, and `CausalReasoner` in `reasoning/interfaces.py` declared
no methods -- the same starting point Phase 4's `reasoning/temporal/` had.
`Relationship` (core/relationships/model.py) has generic directed-edge
fields that *could* host causal-graph edges, but no causal-specific
vocabulary or graph traversal; `Hypothesis`/`Evidence`/`Claim`
(transintelligence/epistemic/) are simple, low-logic dataclasses with no
causal-graph awareness. This module is the first place any of that
becomes operational.

`CausalGraph` implements:

- d-separation via explicit path enumeration and blocking checks (Verma,
  Pearl, *Causal Networks: Semantics and Expressiveness*, UAI 1988) --
  chosen over the equivalent moralized-ancestral-graph reformulation for
  the same reason CUSUM was chosen over a full Bayesian changepoint
  treatment in Phase 4: more directly checkable against the textbook
  three-node canonical cases (chain, fork, collider) this module's own
  tests verify it against, at the cost of being less efficient on large
  graphs -- not a concern at this repo's scale.
- The backdoor criterion (Pearl, *Causal Diagrams for Empirical
  Research*, Biometrika 82(4), 1995) for checking whether a candidate
  adjustment set is valid for estimating a treatment's effect on an
  outcome from purely observational data.
- A small ordinary-least-squares utility for linear effect estimation
  under a candidate adjustment set -- the standard way to operationalize
  "backdoor-adjusted association" for a linear structural causal model
  (the same linearity simplification `BaselineRelativeReasoner` and
  `CUSUMTemporalReasoner` already make elsewhere in this codebase).

Historical/foundational grounding for the surrounding ideas, not
implemented as separate mechanisms here: Simpson, *The Interpretation of
Interaction in Contingency Tables*, JRSS-B 13(2), 1951 (the classic
confounding-reversal phenomenon this module's first experiment
demonstrates a version of); Rubin, *Estimating Causal Effects of
Treatments in Randomized and Nonrandomized Studies*, Journal of
Educational Psychology, 1974 (the potential-outcomes framework, an
alternative formalization of the same causal-effect question this module
answers via graphs instead).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalGraph:
    """A directed acyclic graph of (parent, child) edges. No cycle check
    is performed -- constructing a cyclic graph is a caller error, not
    something this class defends against, matching this repo's general
    preference for trusting internal callers over validating everywhere."""

    edges: tuple[tuple[str, str], ...]

    @property
    def nodes(self) -> set[str]:
        return {n for edge in self.edges for n in edge}

    def parents(self, node: str) -> set[str]:
        return {p for p, c in self.edges if c == node}

    def children(self, node: str) -> set[str]:
        return {c for p, c in self.edges if p == node}

    def ancestors(self, node: str) -> set[str]:
        seen: set[str] = set()
        frontier = list(self.parents(node))
        while frontier:
            p = frontier.pop()
            if p not in seen:
                seen.add(p)
                frontier.extend(self.parents(p))
        return seen

    def descendants(self, node: str) -> set[str]:
        seen: set[str] = set()
        frontier = list(self.children(node))
        while frontier:
            c = frontier.pop()
            if c not in seen:
                seen.add(c)
                frontier.extend(self.children(c))
        return seen

    def _neighbors(self, node: str) -> set[str]:
        return self.parents(node) | self.children(node)

    def _all_simple_paths(self, x: str, y: str) -> list[list[str]]:
        """All simple (no repeated node) paths between x and y, treating
        edges as undirected for traversal purposes -- direction is used
        separately to classify colliders once a path is found."""
        paths: list[list[str]] = []

        def dfs(current: str, target: str, visited: list[str]) -> None:
            if current == target:
                paths.append(list(visited))
                return
            for nxt in self._neighbors(current):
                if nxt not in visited:
                    visited.append(nxt)
                    dfs(nxt, target, visited)
                    visited.pop()

        dfs(x, y, [x])
        return paths

    def _is_collider_on_path(self, path: list[str], i: int) -> bool:
        """Node path[i] (0 < i < len(path)-1) is a collider on this path
        if both adjacent path edges point into it: prev->path[i] and
        next->path[i] both exist as directed edges."""
        prev_node, node, next_node = path[i - 1], path[i], path[i + 1]
        return (prev_node, node) in self.edges and (next_node, node) in self.edges

    def _is_path_blocked(self, path: list[str], conditioning_set: set[str]) -> bool:
        """A path is blocked iff it contains at least one non-collider
        node in `conditioning_set`, or at least one collider node with
        neither itself nor any descendant in `conditioning_set`."""
        for i in range(1, len(path) - 1):
            node = path[i]
            if self._is_collider_on_path(path, i):
                if node not in conditioning_set and not (self.descendants(node) & conditioning_set):
                    return True  # collider blocks: neither it nor a descendant is conditioned on
            elif node in conditioning_set:
                return True  # non-collider in Z blocks the path
        return False

    def d_separated(self, x: str, y: str, conditioning_set: set[str] | None = None) -> bool:
        """True iff every simple path between x and y is blocked given
        conditioning_set -- the standard d-separation criterion (Verma &
        Pearl 1988)."""
        z = conditioning_set or set()
        return all(self._is_path_blocked(path, z) for path in self._all_simple_paths(x, y))

    def satisfies_backdoor_criterion(self, treatment: str, outcome: str, adjustment_set: set[str]) -> bool:
        """Pearl's backdoor criterion (1995): adjustment_set is valid for
        estimating treatment's effect on outcome iff (a) no node in it is
        a descendant of treatment, and (b) it blocks every backdoor path
        (a path from treatment to outcome whose first edge points INTO
        treatment) from treatment to outcome."""
        if adjustment_set & self.descendants(treatment):
            return False
        for path in self._all_simple_paths(treatment, outcome):
            if len(path) < 2:
                continue
            first_edge_into_treatment = (path[1], treatment) in self.edges
            if first_edge_into_treatment and not self._is_path_blocked(path, adjustment_set):
                return False
        return True


def ordinary_least_squares(features: list[list[float]], targets: list[float]) -> list[float]:
    """Minimal OLS via the normal equations (X^T X) beta = X^T y, solved
    by Gaussian elimination with partial pivoting -- no numpy dependency,
    matching this repo's lightweight-dependencies constraint. `features`
    rows should already include a leading 1.0 for the intercept if one is
    wanted. Intended for the handful of covariates this module's
    experiments use (adjustment sets of size 1-3), not general-purpose
    regression at scale."""
    n_rows = len(features)
    n_cols = len(features[0])
    if any(len(row) != n_cols for row in features):
        raise ValueError("all feature rows must have the same length")
    if len(targets) != n_rows:
        raise ValueError("features and targets must have the same number of rows")

    xtx = [[sum(features[k][i] * features[k][j] for k in range(n_rows)) for j in range(n_cols)] for i in range(n_cols)]
    xty = [sum(features[k][i] * targets[k] for k in range(n_rows)) for i in range(n_cols)]

    augmented = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(n_cols):
        pivot_row = max(range(col, n_cols), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("singular design matrix -- collinear or degenerate features")
        augmented[col] = [v / pivot for v in augmented[col]]
        for r in range(n_cols):
            if r != col:
                factor = augmented[r][col]
                augmented[r] = [a - factor * b for a, b in zip(augmented[r], augmented[col])]
    return [augmented[i][n_cols] for i in range(n_cols)]
