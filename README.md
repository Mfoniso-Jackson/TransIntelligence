# TransIntelligence

TransIntelligence is a domain-agnostic intelligence kernel for representing, contextualizing, reasoning over, and acting upon structured environments.

This repository currently implements the first milestone: typed primitives for entities, relationships, observations, states, contexts, reference frames, simple geometric reasoning, baseline relative reasoning, evidence-aware claims, simple memory, and a minimal simulated agent loop.

## Implemented

- Core representation: `Entity`, `Relationship`, `Observation`, `State`, `Context`.
- Reference-frame model with composition and comparison.
- Reasoning protocols plus deterministic geometric and relative baselines.
- Temporal reasoning: regime-change detection, multi-key tracking, and dynamic time warping over `StateHistory` (`transintelligence/reasoning/temporal/`).
- Causal reasoning: causal graphs, d-separation, the backdoor criterion for confounding-bias correction, PC-style causal discovery (skeleton recovery, collider orientation, and Meek's R1-R3 edge-orientation-propagation rules), and identification under an unobserved confounder via two-stage least squares or front-door adjustment (`transintelligence/reasoning/causal/`).
- Counterfactual reasoning: per-unit "what would Y have been had X been different" queries via Pearl's abduction-action-prediction procedure, for linear or nonlinear structural equations (`transintelligence/reasoning/counterfactual/`).
- World models: learned per-action forward dynamics (`LinearDynamicsModel`) for 1-step-lookahead planning (`transintelligence/world_models/`), and matched synthetic control environments including a genuinely nonlinear one (`environments/transworld/resource_control_env.py`, `environments/transworld/delayed_control_env.py`, `environments/transworld/nonlinear_control_env.py`).
- Planning: domain-agnostic receding-horizon (Model Predictive Control-style) multi-step planning over any learned or oracle transition model, with optional beam search for scalable search (`RecedingHorizonPlanner`, `transintelligence/planning/`).
- Combining world models with regime-change detection: `CUSUMTemporalReasoner` (Phase 4) detecting a mid-experiment shift in a learned dynamics model's own residuals, triggering data-discarding adaptation -- tested with both single-step (`experiments/exp13_regime_shift_world_model/`) and multi-step (`experiments/exp16_regime_shift_multistep_planning/`) planning (`environments/transworld/regime_shift_control_env.py`, `environments/transworld/delayed_regime_shift_env.py`).
- Simulation: `MonteCarloSimulator` (`transintelligence/simulation/`) -- comparing the full multi-step outcome distribution of two given candidate policies via stochastic rollouts, rather than predicting one state or searching for one best action.
- Epistemic models: `Claim`, `Evidence`, `CounterEvidence`, `Hypothesis`, `Confidence`, `Source`.
- In-memory episodic/semantic/strategic storage namespaces.
- Minimal `KernelAgent` loop that stores simulated steps.
- Finance and knowledge examples showing the same primitives across domains.

## Planned

Predictive modeling beyond linear dynamics, verification, richer memory, persistence, and production domain adapters are intentionally left behind replaceable interfaces (`Verifier` remains unbuilt; `Predictor`, `Planner`, and now `Simulator` are filled).

## Research

See [docs/master-context.md](docs/master-context.md) for the founding research brief (vision, phased roadmap, working style) this program operationalizes.

Seventeen falsifiable experiments from [docs/research-agenda.md](docs/research-agenda.md) have run, closing out Phase 5 (causal/counterfactual reasoning) and Phase 6 (world models). Short version: reference-frame conditioning helps within a bounded regime, and every result that looked clean on first pass got smaller once the control that could have killed it was actually run — except four: one whose calibration approach independently validated a fix an earlier experiment's failure had named but not built, two (the backdoor criterion for confounding bias, and per-unit counterfactual recovery built on top of it) whose confound controls confirmed the mechanism with textbook clarity, and one (causal discovery) whose negative control ruled out the obvious way its clean numbers could have been a statistical-power illusion. A ninth experiment (instrumental variables and front-door adjustment) is the exception to the exception: its two positive results each came with a matching demonstration of the mechanism's own theory-predicted failure mode, built in from the start rather than found afterward. A tenth confirmed the linearity simplification every causal/counterfactual mechanism in this program made from the start matters exactly where theory says it should: not at all for counterfactual abduction, substantially for linear effect estimation — an eleventh found the same lesson recurring in a completely different setting (planning, not estimation), then a follow-up confirmed it directly by showing a correctly-specified nonlinear baseline closes the gap. A twelfth found a real but modest planning-horizon advantage, isolated from model quality by a 2x2 design and reported at its honest size rather than talked up. A thirteenth combined two already-verified mechanisms (change detection, learned dynamics) and found the honest result is conditional: no adaptation benefit under a mild regime shift (a real null result, investigated and explained rather than smoothed over), but a clear, confound-controlled benefit under a severe one. A fourteenth set out to test a small, expected scalability fix (beam search vs. exhaustive search) and, by investigating a result that looked wrong instead of reporting it, found beam search isn't just cheaper — it's measurably more robust to a genuine receding-horizon oscillation pathology exhaustive search's terminal-only scoring is vulnerable to. A fifteenth closed out a three-object arc traced across the whole program (causal effect estimation, single-step value estimation, and now world-model dynamics): a linear fit cannot represent a nonlinearity, and the gap only shows up once the misspecification is severe enough to actually change a decision. A sixteenth, a second synthesis experiment, independently replicated the fourteenth's oscillation pathology in an unrelated environment, then found something new only visible in combination: learned multi-step planning persistently underperforms learned single-step planning once combined with regime-adaptation, with the obvious "reduced exploration" explanation checked directly and refuted. A seventeenth closed out Phase 6 with a validation rather than a new claim: a newly-built Monte Carlo policy comparator (`MonteCarloSimulator`), using short stochastic rollouts instead of full multi-episode environment averages, independently recovered the sixteenth's exact ranking in 58 of 60 comparisons, with a built-in sensitivity control confirming the rollout count itself was doing real work. See [docs/findings.md](docs/findings.md) for the summary, or [docs/related-work.md](docs/related-work.md) for how these claims relate to prior art.

## Quick start

```bash
python -m pytest
python examples/finance_demo.py
python examples/knowledge_demo.py
```
