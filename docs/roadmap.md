# Roadmap

## Current milestone

A working core that represents entities, relationships, observations, states, contexts, and reference frames, then performs basic geometric and relative reasoning with traceable results. Seven falsifiable experiments have run against the reference-frame-conditioning hypothesis and related claims (see `docs/research-agenda.md`, summarized in `docs/findings.md`).

Temporal reasoning has also started: `transintelligence/reasoning/temporal/` implements regime-change detection, multi-key joint detection, and dynamic time warping (`CUSUMTemporalReasoner`, docs/research-agenda.md #7b) — the first reusable kernel primitive from this research program, rather than an experiment-only script.

Causal reasoning has also started: `transintelligence/reasoning/causal/` implements d-separation and the backdoor criterion (`CausalGraph`, docs/research-agenda.md #7c). Experiment 6 demonstrated that the graph-theoretic criterion, computed with no data at all, exactly predicts which covariate adjustments remove confounding bias — the second reusable kernel primitive from this research program.

Counterfactual reasoning has also started: `transintelligence/reasoning/counterfactual/` implements Pearl's abduction-action-prediction procedure (`StructuralCausalModel`, docs/research-agenda.md #7d) for per-unit "what would Y have been had X been different" queries. Experiment 7 found exact per-unit recovery with true coefficients and a ~9x advantage over a naive plug-in shortcut with estimated ones — the third reusable kernel primitive, and the last previously-empty reasoning protocol stub now filled.

## Next milestone

Add richer observation querying, reference-frame validation, domain adapter examples for growth and property, and structured verifier outputs. Causal reasoning's remaining gaps: front-door adjustment and instrumental variables (only the backdoor criterion is implemented), causal discovery (structure is currently given, not inferred), and nonlinear structural equations (both `reasoning/causal/` and `reasoning/counterfactual/` currently assume linearity).

## Later milestones

- ~~Temporal trajectories and event reasoning.~~ Started — see above; event
  reasoning remains.
- ~~Causal and counterfactual engines.~~ Both started — see above. Every
  reasoning protocol stub that existed before Phase 4 now has a real
  implementation except `Predictor`, `Simulator`, `Planner`, `Verifier`
  (`EvidenceVerifier` in `transintelligence/verification/` is a separate,
  earlier, unrelated implementation).
- Predictor, simulator, and planner implementations.
- Persistent storage adapters, eventually PostgreSQL/pgvector.
- External model provider adapters behind interfaces.
