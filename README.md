# TransIntelligence

TransIntelligence is a domain-agnostic intelligence kernel for representing, contextualizing, reasoning over, and acting upon structured environments.

This repository currently implements the first milestone: typed primitives for entities, relationships, observations, states, contexts, reference frames, simple geometric reasoning, baseline relative reasoning, evidence-aware claims, simple memory, and a minimal simulated agent loop.

## Implemented

- Core representation: `Entity`, `Relationship`, `Observation`, `State`, `Context`.
- Reference-frame model with composition and comparison.
- Reasoning protocols plus deterministic geometric and relative baselines.
- Epistemic models: `Claim`, `Evidence`, `CounterEvidence`, `Hypothesis`, `Confidence`, `Source`.
- In-memory episodic/semantic/strategic storage namespaces.
- Minimal `KernelAgent` loop that stores simulated steps.
- Finance and knowledge examples showing the same primitives across domains.

## Planned

Temporal, causal, counterfactual, predictive, simulation, planning, verification, richer memory, persistence, and production domain adapters are intentionally left behind replaceable interfaces.

## Quick start

```bash
python -m pytest
python examples/finance_demo.py
python examples/knowledge_demo.py
```
