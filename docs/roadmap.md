# Roadmap

## Current milestone

A working core that represents entities, relationships, observations, states, contexts, and reference frames, then performs basic geometric and relative reasoning with traceable results. Six falsifiable experiments have run against the reference-frame-conditioning hypothesis and related claims (see `docs/research-agenda.md`, summarized in `docs/findings.md`).

Temporal reasoning has also started: `transintelligence/reasoning/temporal/` implements regime-change detection, multi-key joint detection, and dynamic time warping (`CUSUMTemporalReasoner`, docs/research-agenda.md #7b) — the first reusable kernel primitive from this research program, rather than an experiment-only script.

Causal reasoning has also started: `transintelligence/reasoning/causal/` implements d-separation and the backdoor criterion (`CausalGraph`, docs/research-agenda.md #7c). Experiment 6 demonstrated that the graph-theoretic criterion, computed with no data at all, exactly predicts which covariate adjustments remove confounding bias — the second reusable kernel primitive from this research program.

## Next milestone

Add richer observation querying, reference-frame validation, domain adapter examples for growth and property, and structured verifier outputs. Extend causal reasoning toward counterfactual queries (`CounterfactualReasoner`, still an empty stub) — per-unit "what would Y have been had X been different," not just population-level adjustment.

## Later milestones

- ~~Temporal trajectories and event reasoning.~~ Started — see above; event
  reasoning remains.
- ~~Causal and counterfactual engines.~~ Causal started — see above.
  Counterfactual (per-unit queries, `CounterfactualReasoner`) remains, and
  is now the most natural next Phase 5 experiment.
- Predictor, simulator, and planner implementations.
- Persistent storage adapters, eventually PostgreSQL/pgvector.
- External model provider adapters behind interfaces.
