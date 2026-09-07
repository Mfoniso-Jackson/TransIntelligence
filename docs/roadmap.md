# Roadmap

## Current milestone

A working core that represents entities, relationships, observations, states, contexts, and reference frames, then performs basic geometric and relative reasoning with traceable results. Five falsifiable experiments have run against the reference-frame-conditioning hypothesis and related claims (see `docs/research-agenda.md`, summarized in `docs/findings.md`).

Temporal reasoning has also started: `transintelligence/reasoning/temporal/` now implements regime-change detection (`CUSUMTemporalReasoner`, docs/research-agenda.md #7b) — the first reusable kernel primitive from this research program, rather than an experiment-only script.

## Next milestone

Add richer observation querying, reference-frame validation, domain adapter examples for growth and property, and structured verifier outputs. Extend temporal reasoning beyond change detection: `regime_segments()` per-segment accuracy under noise, and temporal comparison of trajectories (not just point-to-point state diffs).

## Later milestones

- ~~Temporal trajectories and event reasoning.~~ Started — see above; event
  reasoning and richer trajectory comparison (e.g. dynamic time warping,
  docs/related-work.md §9a) remain.
- Causal and counterfactual engines.
- Predictor, simulator, and planner implementations.
- Persistent storage adapters, eventually PostgreSQL/pgvector.
- External model provider adapters behind interfaces.
