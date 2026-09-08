# Roadmap

## Current milestone

A working core that represents entities, relationships, observations, states, contexts, and reference frames, then performs basic geometric and relative reasoning with traceable results. Nine falsifiable experiments have run against the reference-frame-conditioning hypothesis and related claims (see `docs/research-agenda.md`, summarized in `docs/findings.md`).

Temporal reasoning has also started: `transintelligence/reasoning/temporal/` implements regime-change detection, multi-key joint detection, and dynamic time warping (`CUSUMTemporalReasoner`, docs/research-agenda.md #7b) — the first reusable kernel primitive from this research program, rather than an experiment-only script.

Causal reasoning has also started: `transintelligence/reasoning/causal/` implements d-separation, the backdoor criterion, PC-style causal discovery, two-stage least squares, and front-door adjustment (`CausalGraph`, `discover_skeleton`, `orient_colliders`, `two_stage_least_squares`, `front_door_adjustment`, docs/research-agenda.md #7c, #7e, #7f). Experiment 6 demonstrated that the graph-theoretic criterion, computed with no data at all, exactly predicts which covariate adjustments remove confounding bias; experiment 8 demonstrated that the graph itself doesn't have to be given — skeleton recovery and collider orientation reconstruct it from data, with a negative control ruling out statistical-power artifacts; experiment 9 demonstrated that even a confounder that's never observed at all doesn't block identification, given a valid instrument or mediator, though both of those strategies fail in their own theory-predicted ways once their assumptions don't hold — the second, fourth, and fifth reusable kernel primitives from this research program.

Counterfactual reasoning has also started: `transintelligence/reasoning/counterfactual/` implements Pearl's abduction-action-prediction procedure (`StructuralCausalModel`, docs/research-agenda.md #7d) for per-unit "what would Y have been had X been different" queries. Experiment 7 found exact per-unit recovery with true coefficients and a ~9x advantage over a naive plug-in shortcut with estimated ones — the third reusable kernel primitive, and the last previously-empty reasoning protocol stub now filled.

## Next milestone

Add richer observation querying, reference-frame validation, domain adapter examples for growth and property, and structured verifier outputs. Causal reasoning's remaining gaps: Meek's further orientation-propagation rules for causal discovery (only skeleton recovery plus collider orientation are implemented), and nonlinear structural equations (both `reasoning/causal/` and `reasoning/counterfactual/` currently assume linearity) -- backdoor adjustment, causal discovery, instrumental variables, and front-door adjustment are all now implemented.

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
