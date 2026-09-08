# Experiment 14 results — beam search for `RecedingHorizonPlanner`

Ran: `PYTHONPATH=. python experiments/exp14_beam_search_planning/run.py`.
15 random starting states × 5-step receding-horizon rollouts (replanning
every step) per condition. See `docs/research-agenda.md` #7k for the
hypothesis and `docs/related-work.md` §3h for the grounding theory
(Lowerre 1976, beam search).

This started as the expected, small, well-scoped follow-up: does beam
search let `RecedingHorizonPlanner` (experiments 12-13) scale past its
exhaustive `O(actions^depth)` search, at a small quality cost? It ended
up finding something more interesting, by investigating a result that
looked wrong instead of reporting it. See the module docstring in
`experiments/exp14_beam_search_planning/run.py` for the full trace of
two real methodological bugs caught along the way — summarized here.

## The actual finding: beam search isn't just cheaper, it's more robust

| depth | method | mean score | converged (of 15) | mean `transition_fn` calls | call reduction |
|---|---|---|---|---|---|
| 2 | exhaustive | -0.0170 | 13/15 | 72.0 | 1.0x |
| 2 | beam_width=2 | -0.0045 | **15/15** | 18.0 | 4.0x |
| 2 | beam_width=3 | -0.0168 | 13/15 | 24.0 | 3.0x |
| 2 | beam_width=4 | -0.0170 | 13/15 | 30.0 | 2.4x |
| 3 | exhaustive | -1.9568 | **2/15** | 648.0 | 1.0x |
| 3 | beam_width=2 | -0.0045 | **15/15** | 30.0 | 21.6x |
| 3 | beam_width=3 | -0.0168 | 13/15 | 42.0 | 15.4x |
| 3 | beam_width=4 | -0.0170 | 13/15 | 54.0 | 12.0x |

**At depth 3, exhaustive search converges to (and stays at) the target
in only 2 of 15 starting states — while every beam width tested
converges in 13-15 of 15, using 12-22x fewer `transition_fn` calls.**
Deeper naive lookahead makes exhaustive search *worse* at reaching the
target, not better (13/15 at depth 2 → 2/15 at depth 3): more lookahead
gives it more flexibility to find a plan whose *promised* final position
looks good on paper, disconnected from whether actually executing and
replanning it converges anywhere.

## Why: what exhaustive search optimizes vs. what receding-horizon control needs

Traced one misbehaving state step by step rather than re-averaging and
hoping the anomaly would wash out. Exhaustive search's chosen policy
entered a **persistent oscillation** — position stuck away from target,
`pending` alternating sign every single step, forever:

```
state 1 start=(-0.79, -2.41): final_score=-6.2485
   step 1: position=-2.4997 pending=-1.0000
   step 2: position=-2.4997 pending= 1.0000
   step 3: position=-2.4997 pending=-1.0000
   step 4: position=-2.4997 pending= 1.0000
   step 5: position=-2.4997 pending=-1.0000   <- never converges
```

Beam search (`beam_width=2`), from the exact same starting state,
converges to the target and stays there:

```
   step 1: position=-1.4997 pending= 1.0000
   step 2: position= 0.0003 pending= 2.0000
   step 3: position= 0.0003 pending=-2.0000
   step 4: position= 0.0003 pending= 2.0000
   step 5: position= 0.0003 pending=-2.0000   <- position holds at target
```

**The mechanism**: exhaustive search scores only the *final* simulated
state after the full `depth`-step lookahead — indifferent to the path
taken to get there. It can select a sequence whose promised final
position looks optimal, while its *first* action (the only one actually
executed before the receding-horizon protocol replans from scratch)
sets up a self-reinforcing overshoot-correct-overshoot cycle. Beam
search, by contrast, scores every *intermediate* partial state during
its own expansion and prunes candidates whose partial trajectory already
looks bad — an incidental but real bias toward monotonic progress that
happens to make it robust to exactly this receding-horizon pathology,
not just cheaper.

## Two real bugs this experiment's own early runs caught

1. **Single-step evaluation.** The first version scored only one
   executed step's immediate outcome, not the multi-step rollout the
   search was actually optimizing for — showing beam search
   "beating" exhaustive, which is impossible for the objective a
   single-step evaluation doesn't actually measure. Fixed by scoring a
   full 5-step receding-horizon rollout with replanning at every step
   (Richalet et al. 1978's actual protocol, and how experiment 12
   evaluated planning quality).
2. **Tie-breaking degeneracy in a fine action grid.** A first attempt
   used a finer, 21-value action vocabulary; checked directly, one state
   at depth 3 had 19 different 3-action sequences all scoring exactly
   0.0, with first actions ranging from -2.0 to 0.0 — and exhaustive
   search's tie-breaking (`score > best_score`, strict, `self.actions`
   ascending) always favored the most extreme one. Fixed by reusing
   experiment 12's original, widely-spaced 6-action vocabulary, checked
   to have far fewer ties (1 and 2 tied sequences at depths 2 and 3,
   respectively).

Even after both fixes, the oscillation pathology above remained — it
isn't an artifact of either bug, it's the real finding.

## What this establishes

`RecedingHorizonPlanner.choose_action`'s `beam_width` parameter was
verified against exhaustive search directly before being trusted for
anything: `beam_width` equal to the action count reproduces exhaustive
search's choice exactly at every depth tested, and `transition_fn` call
counts match hand-computed predictions exactly (`tests/test_planning.py`,
before this experiment). This experiment establishes that beam search's
practical advantage over exhaustive search, in a receding-horizon
control setting specifically, is not purely a scalability tradeoff —
its incremental, per-step scoring is a better match for how
receding-horizon control actually gets executed (one action at a time,
replanned from the true resulting state) than exhaustive search's
terminal-only objective is.

## What this does not establish

- **A general claim about beam search vs. exhaustive search's
  *planning-time* objective** — at a fixed depth, exhaustive search
  really does find the sequence with the best *simulated* final score;
  the finding here is specifically about what happens once only the
  first action is executed and the rest is discarded (the receding-
  horizon protocol), not a claim that exhaustive search's own stated
  objective is wrong.
- **Whether a smarter tie-breaking rule (e.g. preferring minimal-effort
  actions among ties) would close the gap** — not attempted; the fix
  applied here removed the pathological action vocabulary rather than
  fixing exhaustive search's tie-breaking directly.
- **Deeper depths were not tested with ground truth** — depth 6 only
  reports beam search's own results (exhaustive is intractable there);
  whether exhaustive search's convergence failure gets worse, better, or
  plateaus beyond depth 3 is unknown.
- **A single environment (the first-order-lag control task from
  experiment 12)** — whether this oscillation pathology is specific to
  this dynamics structure or general to receding-horizon control with
  terminal-only scoring more broadly is not tested here.
