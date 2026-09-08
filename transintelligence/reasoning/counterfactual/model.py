"""Baseline counterfactual reasoning: per-unit "what would Y have been had
X been different" queries over a linear structural causal model (Phase 5,
docs/research-agenda.md #7d).

`transintelligence/reasoning/counterfactual/` was, before this, a
single-line docstring stub, and `CounterfactualReasoner` in
`reasoning/interfaces.py` declared no methods -- the same starting point
`reasoning/causal/` had before Experiment 6. That module answers a
population-level question ("does X affect Y on average, adjusting for
confounders"); this one answers a per-unit question ("what would THIS
specific unit's Y have been, had its X been different") -- Pearl's third
rung of the causal hierarchy, and a genuinely different capability, not
just a bigger version of the backdoor-adjustment machinery.

`StructuralCausalModel` implements Pearl's three-step counterfactual
procedure -- abduction, action, prediction (Pearl, Glymour, Jewell,
*Causal Inference in Statistics: A Primer*, Wiley, 2016; the underlying
computational treatment traces to Balke, Pearl, *Counterfactual
Probabilities: Computational Methods, Bounds and Applications*, UAI 1994)
-- specialized to linear structural equations with additive exogenous
noise, the same linearity simplification `BaselineRelativeReasoner`,
`CUSUMTemporalReasoner`, and `reasoning/causal/`'s effect estimation
already make elsewhere in this codebase. For a linear+additive-noise SCM,
abduction has a closed form (the residual of each node's equation) rather
than requiring general inference -- the smallest mechanism that could
produce a falsifiable result, not a claim to handle nonlinear or
non-additive-noise SCMs (Pearl's general theory covers those; this
doesn't).
"""
from __future__ import annotations

from dataclasses import dataclass

from transintelligence.reasoning.causal import CausalGraph


@dataclass(frozen=True)
class StructuralEquation:
    """node = intercept + sum(coefficients[parent] * value[parent]) + exogenous_noise."""

    coefficients: dict[str, float]
    intercept: float = 0.0

    def predict(self, parent_values: dict[str, float]) -> float:
        return self.intercept + sum(coef * parent_values[p] for p, coef in self.coefficients.items())


@dataclass(frozen=True)
class StructuralCausalModel:
    """A `CausalGraph` plus one `StructuralEquation` per non-root node.
    Root nodes (no entry in `equations`) are treated as exogenous --
    their observed value *is* their noise term, since nothing generates
    them structurally within this model."""

    graph: CausalGraph
    equations: dict[str, StructuralEquation]

    def _topological_order(self) -> list[str]:
        nodes = self.graph.nodes | set(self.equations.keys())
        in_degree = {n: len(self.graph.parents(n) & nodes) for n in nodes}
        ready = [n for n in nodes if in_degree[n] == 0]
        order: list[str] = []
        while ready:
            n = ready.pop()
            order.append(n)
            for c in self.graph.children(n):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    ready.append(c)
        if len(order) != len(nodes):
            raise ValueError("graph is not acyclic, or a node in `equations` has no edges in `graph`")
        return order

    def abduct(self, observed: dict[str, float]) -> dict[str, float]:
        """Infer each node's exogenous noise term consistent with a fully
        observed unit -- the closed-form residual of its structural
        equation for non-root nodes, or the observed value itself for
        exogenous roots."""
        noise: dict[str, float] = {}
        for node in self._topological_order():
            if node not in self.equations:
                noise[node] = observed[node]
            else:
                eq = self.equations[node]
                predicted = eq.predict({p: observed[p] for p in eq.coefficients})
                noise[node] = observed[node] - predicted
        return noise

    def counterfactual(self, observed: dict[str, float], intervention: dict[str, float]) -> dict[str, float]:
        """Pearl's abduction-action-prediction procedure: infer this
        unit's exogenous noise from `observed` (abduction), fix the
        intervened nodes to their new values (action), then recompute
        every other node in topological order using its own structural
        equation and its *own* inferred noise (prediction) -- so the
        result reflects what would have happened to THIS unit, not the
        population average, had `intervention` held instead of the
        observed values."""
        noise = self.abduct(observed)
        result = dict(observed)
        result.update(intervention)
        for node in self._topological_order():
            if node in intervention:
                continue
            if node not in self.equations:
                result[node] = noise[node]
                continue
            eq = self.equations[node]
            predicted = eq.predict({p: result[p] for p in eq.coefficients})
            result[node] = predicted + noise[node]
        return result
