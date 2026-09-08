# TransIntelligence

TransIntelligence is a domain-agnostic intelligence kernel for representing, contextualizing, reasoning over, and acting upon structured environments.

This repository currently implements the first milestone: typed primitives for entities, relationships, observations, states, contexts, reference frames, simple geometric reasoning, baseline relative reasoning, evidence-aware claims, simple memory, and a minimal simulated agent loop.

## Implemented

- Core representation: `Entity`, `Relationship`, `Observation`, `State`, `Context`.
- Reference-frame model with composition and comparison.
- Reasoning protocols plus deterministic geometric and relative baselines.
- Temporal reasoning: regime-change detection, multi-key tracking, and dynamic time warping over `StateHistory` (`transintelligence/reasoning/temporal/`).
- Causal reasoning: causal graphs, d-separation, the backdoor criterion for confounding-bias correction, PC-style causal discovery (skeleton recovery plus collider orientation), and identification under an unobserved confounder via two-stage least squares or front-door adjustment (`transintelligence/reasoning/causal/`).
- Counterfactual reasoning: per-unit "what would Y have been had X been different" queries via Pearl's abduction-action-prediction procedure (`transintelligence/reasoning/counterfactual/`).
- Epistemic models: `Claim`, `Evidence`, `CounterEvidence`, `Hypothesis`, `Confidence`, `Source`.
- In-memory episodic/semantic/strategic storage namespaces.
- Minimal `KernelAgent` loop that stores simulated steps.
- Finance and knowledge examples showing the same primitives across domains.

## Planned

Predictive, simulation, planning, verification, richer memory, persistence, and production domain adapters are intentionally left behind replaceable interfaces.

## Research

Nine falsifiable experiments from [docs/research-agenda.md](docs/research-agenda.md) have run. Short version: reference-frame conditioning helps within a bounded regime, and every result that looked clean on first pass got smaller once the control that could have killed it was actually run — except four: one whose calibration approach independently validated a fix an earlier experiment's failure had named but not built, two (the backdoor criterion for confounding bias, and per-unit counterfactual recovery built on top of it) whose confound controls confirmed the mechanism with textbook clarity, and one (causal discovery) whose negative control ruled out the obvious way its clean numbers could have been a statistical-power illusion. A ninth experiment (instrumental variables and front-door adjustment) is the exception to the exception: its two positive results each came with a matching demonstration of the mechanism's own theory-predicted failure mode, built in from the start rather than found afterward. See [docs/findings.md](docs/findings.md) for the summary, or [docs/related-work.md](docs/related-work.md) for how these claims relate to prior art.

## Quick start

```bash
python -m pytest
python examples/finance_demo.py
python examples/knowledge_demo.py
```
