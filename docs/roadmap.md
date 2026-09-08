# Roadmap

## Current milestone

A working core that represents entities, relationships, observations, states, contexts, and reference frames, then performs basic geometric and relative reasoning with traceable results. Twelve falsifiable experiments have run against the reference-frame-conditioning hypothesis and related claims (see `docs/research-agenda.md`, summarized in `docs/findings.md`).

Temporal reasoning has also started: `transintelligence/reasoning/temporal/` implements regime-change detection, multi-key joint detection, and dynamic time warping (`CUSUMTemporalReasoner`, docs/research-agenda.md #7b) — the first reusable kernel primitive from this research program, rather than an experiment-only script.

Causal reasoning has also started: `transintelligence/reasoning/causal/` implements d-separation, the backdoor criterion, PC-style causal discovery (including Meek's R1-R3 edge-orientation-propagation rules), two-stage least squares, and front-door adjustment (`CausalGraph`, `discover_skeleton`, `orient_colliders`, `apply_meek_rules`, `two_stage_least_squares`, `front_door_adjustment`, docs/research-agenda.md #7c, #7e, #7f). Experiment 6 demonstrated that the graph-theoretic criterion, computed with no data at all, exactly predicts which covariate adjustments remove confounding bias; experiment 8 demonstrated that the graph itself doesn't have to be given — skeleton recovery and collider orientation reconstruct it from data, with a negative control ruling out statistical-power artifacts, and a follow-up showed Meek's rules extend a hard 0.500 recall ceiling to full recovery on identifiable graphs without ever orienting wrongly; experiment 9 demonstrated that even a confounder that's never observed at all doesn't block identification, given a valid instrument or mediator, though both of those strategies fail in their own theory-predicted ways once their assumptions don't hold — the second, fourth, and fifth reusable kernel primitives from this research program.

Counterfactual reasoning has also started: `transintelligence/reasoning/counterfactual/` implements Pearl's abduction-action-prediction procedure (`StructuralCausalModel`, docs/research-agenda.md #7d) for per-unit "what would Y have been had X been different" queries, for linear or nonlinear structural equations (`StructuralEquation.nonlinear_fn`, docs/research-agenda.md #7g). Experiment 7 found exact per-unit recovery with true coefficients and a ~9x advantage over a naive plug-in shortcut with estimated ones — the third reusable kernel primitive, and the last previously-empty reasoning protocol stub now filled. Experiment 10 confirmed abduction stays exact under nonlinearity with no code changes to `abduct()`/`counterfactual()` at all, while `reasoning/causal/`'s linear effect estimation is substantially biased there and cannot represent a heterogeneous (unit-dependent) effect at all.

Phase 6 (World Models, `docs/master-context.md` §13/§19) has now started: `transintelligence/world_models/` implements `LinearDynamicsModel`, a learned per-action forward dynamics model (docs/research-agenda.md #7h), filling the `Predictor` protocol stub. `environments/transworld/resource_control_env.py` is the first environment with genuine state→action→consequence dynamics (`FrameSwitchEnv`, used through experiment 4, is a stateless bandit — the agent's choice never changes a persistent world state). Experiment 11 found a world-model-planning agent matches an oracle's performance ceiling almost exactly, while an equally state-aware, identically-tooled model-free baseline falls far short — the same "a linear fit can't represent a relationship with a peak" lesson experiment 10 found for effect estimation, now shown in a planning setting; a follow-up confirmed the mechanism directly by showing a correctly-specified nonlinear baseline closes almost the entire gap — the fourth reusable kernel primitive from this research program's causal/world-model family (fifth counting `CUSUMTemporalReasoner`).

Multi-step planning has also started: `environments/transworld/delayed_control_env.py` is the first environment with genuine multi-step credit assignment (episodes persist across steps, reward only at the final step, unlike experiment 11's fresh-state-per-trial design). Experiment 12 found a real, consistent, but honestly modest planning-horizon advantage over greedy 1-step lookahead — isolated from model-quality via a 2x2 factorial design (docs/research-agenda.md #7i). Its planning logic has since been generalized into `transintelligence/planning/`'s `RecedingHorizonPlanner` — a domain-agnostic kernel primitive (needs only a `transition_fn`/`score_fn`, no particular state representation or action vocabulary) filling the `Planner` protocol stub, refactored in after experiment 12 shipped and verified to produce bit-for-bit identical results — the sixth reusable kernel primitive from this research program.

## Next milestone

Add richer observation querying, reference-frame validation, domain adapter examples for growth and property, and structured verifier outputs. Causal reasoning's stated gaps are now all closed: backdoor adjustment, causal discovery (skeleton recovery, collider orientation, and Meek's R1-R3 edge-orientation-propagation rules -- R4 is provably inapplicable without a background-knowledge mechanism this pipeline doesn't have), instrumental variables, front-door adjustment, and nonlinear structural equations (for counterfactual abduction; linear-only effect estimation remains a known, demonstrated limit, not a gap left untested) are all implemented. `Predictor` and `Planner` are now both filled too. Phase 6's remaining gaps: nonlinear dynamics; a non-stationary environment (combining world models with the existing regime-change-detection machinery from Phase 4 is untested); a smarter search strategy for `RecedingHorizonPlanner` (exhaustive enumeration over action sequences won't scale to larger action sets or longer horizons); and `Simulator`, which remains fully unbuilt.

## Later milestones

- ~~Temporal trajectories and event reasoning.~~ Started — see above; event
  reasoning remains.
- ~~Causal and counterfactual engines.~~ Both started — see above. Every
  reasoning protocol stub that existed before Phase 4 now has a real
  implementation except `Simulator`, `Verifier` (`Predictor` and
  `Planner` are now filled too, by Phase 6's `LinearDynamicsModel` and
  `RecedingHorizonPlanner` — `EvidenceVerifier` in
  `transintelligence/verification/` is a separate, earlier, unrelated
  implementation).
- ~~World models (Phase 6).~~ Started — see above; nonlinear dynamics
  remains.
- ~~Multi-step planning/rollouts.~~ Started — see above; a smarter
  search strategy than exhaustive enumeration remains, and `Simulator`
  is still fully unbuilt.
- Persistent storage adapters, eventually PostgreSQL/pgvector.
- External model provider adapters behind interfaces.
