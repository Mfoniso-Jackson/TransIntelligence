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

`two_stage_least_squares` and `front_door_adjustment` add a third
capability: identifying a treatment's causal effect on an outcome when
the backdoor criterion *cannot* be satisfied because a confounder is
unobserved -- the two classic alternative identification strategies for
exactly that situation.

- Two-stage least squares (2SLS), via a valid instrument: the estimator
  traces to Appendix B of Philip G. Wright, *The Tariff on Animal and
  Vegetable Oils*, Macmillan, 1928 (authorship of that appendix is
  historically disputed between Philip and his son Sewall Wright).
- Front-door adjustment, via a fully-mediating observed variable: Pearl,
  *Causal Diagrams for Empirical Research*, Biometrika 82(4), 1995 -- the
  same paper already cited above for the backdoor criterion. The linear
  specialization implemented here (multiply the treatment-to-mediator and
  mediator-to-outcome path coefficients) is the classical path-analysis
  result for chained linear structural equations: Wright, *The Method of
  Path Coefficients*, Annals of Mathematical Statistics 5(3), 161-215,
  1934 -- any correlation in a network of linear sequential relations
  decomposes into a sum of products of coefficients along connecting
  paths, so a two-step mediator chain's total effect is exactly the
  product of its two path coefficients.

`discover_skeleton`/`orient_colliders` add a fourth, later capability:
recovering graph structure from data instead of assuming it's given, via
the constraint-based PC algorithm (Spirtes & Glymour, *An Algorithm for
Fast Recovery of Sparse Causal Graphs*, Social Science Computer Review
9(1), 1991). This implementation deliberately stops after skeleton
recovery and collider (v-structure) orientation -- it does not implement
Meek's further orientation-propagation rules (Meek, *Causal Inference and
Causal Explanation with Background Knowledge*, UAI 1995, pp. 403-410),
which can orient additional edges beyond colliders under acyclicity and
no-new-collider constraints. Skipped deliberately as the smallest
mechanism that can produce a falsifiable result about structure
discovery at all -- the same discipline that chose CUSUM over full
Bayesian changepoint detection in Phase 4.
"""
from __future__ import annotations

import itertools
import math
import statistics
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


def partial_correlation(data: dict[str, list[float]], x: str, y: str, z: set[str]) -> float:
    """Pearson correlation between x and y after linearly regressing each
    on z and taking residuals -- the standard way to test conditional
    independence for approximately-linear-Gaussian data without a
    multivariate-normal library. z=set() reduces to the ordinary Pearson
    correlation of x and y directly."""
    n = len(data[x])
    if not z:
        xs, ys = data[x], data[y]
    else:
        z_cols = sorted(z)
        features = [[1.0] + [data[k][i] for k in z_cols] for i in range(n)]
        beta_x = ordinary_least_squares(features, data[x])
        beta_y = ordinary_least_squares(features, data[y])
        xs = [data[x][i] - sum(b * f for b, f in zip(beta_x, features[i])) for i in range(n)]
        ys = [data[y][i] - sum(b * f for b, f in zip(beta_y, features[i])) for i in range(n)]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
    var_x = sum((a - mean_x) ** 2 for a in xs)
    var_y = sum((b - mean_y) ** 2 for b in ys)
    if var_x < 1e-12 or var_y < 1e-12:
        return 0.0
    return covariance / math.sqrt(var_x * var_y)


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_z_independence_test(r: float, n: int, conditioning_set_size: int, alpha: float = 0.05) -> bool:
    """True iff x and y are judged conditionally independent given a set
    of size `conditioning_set_size`, via Fisher's z-transform of the
    partial correlation `r` -- the independence test the PC algorithm
    (Spirtes & Glymour 1991) uses for linear-Gaussian data. Too few
    residual degrees of freedom to say anything (n - conditioning_set_size
    - 3 <= 0) is treated as "cannot reject independence" rather than
    raising, since the PC skeleton search calls this with conditioning
    sets of growing size and needs a defined answer at every size."""
    r = max(min(r, 1.0 - 1e-10), -1.0 + 1e-10)
    dof = n - conditioning_set_size - 3
    if dof <= 0:
        return True
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(dof)
    p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(z)))
    return p_value > alpha


@dataclass(frozen=True)
class DiscoveredSkeleton:
    """The undirected output of PC's skeleton-recovery phase: which pairs
    remain adjacent, and, for each pair that was separated, the
    conditioning set that made it so -- the latter is exactly what
    `orient_colliders` needs to tell a collider from a chain/fork."""

    undirected_edges: frozenset[frozenset[str]]
    separating_sets: dict[frozenset[str], frozenset[str]]


def discover_skeleton(data: dict[str, list[float]], alpha: float = 0.05, max_conditioning_set_size: int | None = None) -> DiscoveredSkeleton:
    """PC-algorithm skeleton recovery (Spirtes & Glymour 1991): start from
    a complete undirected graph and remove an edge x-y as soon as some
    conditioning set Z, drawn from x's and y's current neighbors, makes
    them conditionally independent -- testing growing conditioning-set
    sizes in order, exactly as the original algorithm does, which is what
    lets it stay a search over *neighbors* rather than all subsets of all
    other variables. `max_conditioning_set_size` defaults to the largest
    conditioning set that could exist (nodes - 2); this repo's graphs are
    small enough (3-5 nodes) that this is never a real constraint."""
    nodes = sorted(data.keys())
    n = len(data[nodes[0]])
    adjacency: dict[str, set[str]] = {a: set(nodes) - {a} for a in nodes}
    separating_sets: dict[frozenset[str], frozenset[str]] = {}
    limit = max_conditioning_set_size if max_conditioning_set_size is not None else len(nodes) - 2

    conditioning_size = 0
    while conditioning_size <= limit:
        for x in nodes:
            for y in sorted(adjacency[x]):
                if y < x:
                    continue
                candidates = (adjacency[x] | adjacency[y]) - {x, y}
                if len(candidates) < conditioning_size:
                    continue
                for z_tuple in itertools.combinations(sorted(candidates), conditioning_size):
                    z_set = set(z_tuple)
                    r = partial_correlation(data, x, y, z_set)
                    if fisher_z_independence_test(r, n, conditioning_size, alpha):
                        adjacency[x].discard(y)
                        adjacency[y].discard(x)
                        separating_sets[frozenset((x, y))] = frozenset(z_set)
                        break
        conditioning_size += 1

    edges = frozenset(frozenset((a, b)) for a in nodes for b in adjacency[a])
    return DiscoveredSkeleton(undirected_edges=edges, separating_sets=separating_sets)


def orient_colliders(skeleton: DiscoveredSkeleton, nodes: list[str]) -> frozenset[tuple[str, str]]:
    """For every unshielded triple x-z-y (x and y both adjacent to z, but
    NOT adjacent to each other) where z is not in the separating set that
    removed the x-y edge, orient both edges into z: x->z<-y (the
    collider/v-structure orientation rule from Spirtes & Glymour 1991).
    Shielded triples -- where x and y are also directly adjacent -- are
    skipped entirely: a direct x-y edge makes the rule inapplicable there,
    which is exactly why experiment 6's confounding graph (X->Y direct,
    plus the X->W<-Y collider) cannot be used to exercise this rule -- its
    collider is shielded, so a separate graph is needed."""
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for edge in skeleton.undirected_edges:
        a, b = tuple(edge)
        adjacency[a].add(b)
        adjacency[b].add(a)

    directed: set[tuple[str, str]] = set()
    for z in nodes:
        neighbors = sorted(adjacency[z])
        for i, x in enumerate(neighbors):
            for y in neighbors[i + 1:]:
                if y in adjacency[x]:
                    continue  # shielded triple: x and y are directly adjacent
                sep_set = skeleton.separating_sets.get(frozenset((x, y)))
                if sep_set is not None and z not in sep_set:
                    directed.add((x, z))
                    directed.add((y, z))
    return frozenset(directed)


def two_stage_least_squares(instrument: list[float], treatment: list[float], outcome: list[float]) -> float:
    """Two-stage least squares (Wright 1928, Appendix B): identifies a
    treatment's causal effect on an outcome using a valid instrument --
    a variable that (a) affects treatment, (b) has no effect on outcome
    except through treatment, and (c) shares no common cause with
    outcome -- even when an *unobserved* confounder makes ordinary
    regression of outcome on treatment biased.

    Stage 1: regress treatment on the instrument; take fitted values.
    Because the instrument is assumed independent of the confounder, the
    *variation in treatment explained by the instrument* is too --
    isolating the confounder-free part of treatment's variation.
    Stage 2: regress outcome on those fitted values instead of the raw
    treatment. Returns the treatment's estimated coefficient (the causal
    effect estimate) from stage 2."""
    stage1_features = [[1.0, z] for z in instrument]
    stage1_coefficients = ordinary_least_squares(stage1_features, treatment)
    fitted_treatment = [stage1_coefficients[0] + stage1_coefficients[1] * z for z in instrument]
    stage2_features = [[1.0, x_hat] for x_hat in fitted_treatment]
    stage2_coefficients = ordinary_least_squares(stage2_features, outcome)
    return stage2_coefficients[1]


def front_door_adjustment(treatment: list[float], mediator: list[float], outcome: list[float]) -> float:
    """Front-door adjustment (Pearl 1995): identifies a treatment's causal
    effect on an outcome via a fully-mediating observed variable, for the
    case where treatment and outcome share an *unobserved* confounder but
    the confounder has no effect on the mediator -- exactly the situation
    where the backdoor criterion cannot be satisfied (the confounder isn't
    in the data to adjust for) but this alternative identification
    strategy still applies.

    Linear-model implementation: the treatment->mediator effect is
    unconfounded (nothing here confounds treatment and mediator), so it's
    just their regression coefficient. The mediator->outcome effect is
    confounded by treatment (mediator depends on treatment, and treatment
    shares the unobserved confounder with outcome), so it's estimated
    adjusting for treatment -- treatment is a valid backdoor-style
    adjustment for the mediator->outcome relationship specifically,
    since it blocks mediator<-treatment<-confounder->outcome. The total
    effect is the product of the two path coefficients (Wright 1934's
    path-analysis result for chained linear relations), not their sum or
    either one alone."""
    xm_features = [[1.0, x] for x in treatment]
    xm_coefficient = ordinary_least_squares(xm_features, mediator)[1]
    my_features = [[1.0, m, x] for m, x in zip(mediator, treatment)]
    my_coefficient = ordinary_least_squares(my_features, outcome)[1]
    return xm_coefficient * my_coefficient
