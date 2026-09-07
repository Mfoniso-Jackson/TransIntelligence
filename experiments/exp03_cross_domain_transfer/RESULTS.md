# Experiment 3 results — minimal cross-domain structural transfer

Ran: `PYTHONPATH=. python experiments/exp03_cross_domain_transfer/run.py`.
10 trials, `LearnedEmbeddingAgent` trained 3000 steps on DOMAIN_A
(finance-shaped), transferred into DOMAIN_B (knowledge-shaped, 3000 steps),
early-window = first 300 steps. See `docs/research-agenda.md` #7 for the
hypothesis and falsification criterion, `docs/related-work.md` #6 for why
Gentner's structure-mapping distinction (1983) is what defines "isomorphic"
here, and `experiments/exp03_cross_domain_transfer/domains.py` for the
actual frame definitions.

`RFAwareAgent` is not used in this experiment: its "knowledge" is just its
`ReferenceFrame` list, trivial to hand over verbatim if domain B uses the
same numeric frames — nothing is *learned* to test transfer of.
`LearnedEmbeddingAgent`'s trained `(w, b)` slot values are the only thing
in this codebase that are genuinely learned from data.

## The naive result (transfer vs. scratch alone) looked clean

| condition | transfer (early) | scratch (early) | transfer (overall) | scratch (overall) |
|---|---|---|---|---|
| isomorphic | 0.893 | 0.675 | 0.884 | 0.843 |
| non_isomorphic | 0.818 | 0.683 | 0.838 | 0.801 |

```
isomorphic       transfer-scratch (early):    mean=+0.218  stdev=0.045  wins=10/10
non_isomorphic   transfer-scratch (early):    mean=+0.134  stdev=0.039  wins=10/10
```

Read on its own, this looks like a pass with a caveat: the advantage is
smaller on the non-isomorphic control (+0.134 vs +0.218) but very much
still present, 10/10 wins either way. **That reading is incomplete, and
reporting it without the next test would overclaim.**

## The missing control: does *any* pre-differentiated start help, regardless of domain?

Added a third starting condition, `random_differentiated_experts()`:
slots initialized at the same rough magnitude domain-A-trained experts
actually converge to (~1-8, inspected directly — not the tiny `N(0,
0.05)` used for `scratch`), but with **no training on any domain at all**.
This isolates "started already differentiated" from "the transferred
values are actually informed by domain A's rules."

| condition | transfer | random_differentiated | scratch |
|---|---|---|---|
| isomorphic (early) | 0.893 | 0.763 | 0.675 |
| non_isomorphic (early) | 0.818 | 0.747 | 0.683 |
| isomorphic (overall) | 0.884 | 0.868 | 0.843 |
| non_isomorphic (overall) | 0.838 | 0.827 | 0.801 |

Decomposing the total `transfer − scratch` gap into a **confound**
(`random_differentiated − scratch`, present regardless of domain match)
and a **structure-specific** component (`transfer − random_differentiated`,
the part actually attributable to domain A's learned values being
informative):

```
                  metric    total (transfer-scratch)   structure-specific (transfer-rand_diff)   confound (rand_diff-scratch)
isomorphic        early     +0.218 (10/10)              +0.131 (10/10)                            +0.088 (7/10)
non_isomorphic    early     +0.134 (10/10)              +0.071 (8/10)                              +0.064 (8/10)
isomorphic        overall   +0.041 (10/10)              +0.016 (9/10)                              +0.026 (10/10)
non_isomorphic    overall   +0.037 (9/10)               +0.010 (6/10)                              +0.026 (8/10)
```

## Finding: a real but partial, and substantially confounded, transfer effect

**A large fraction of the naive "transfer beats scratch" result is a
confound, not evidence of structural transfer.** `random_differentiated`
(no training, just not-near-zero) beats `scratch` by +0.088 (early,
isomorphic) and +0.064 (early, non-isomorphic) — a sizable, domain-
independent effect. The mechanism is straightforward: `scratch`'s slots
all start within `N(0, 0.05)` of each other, so they predict nearly
identically at first and the Bayesian belief filter has no signal to
differentiate on — every non-degenerate warm start, informed or not,
skips that cold-start period. Any transfer claim that doesn't control for
this is measuring "not starting from scratch" more than "structure
transferred."

**Once that confound is subtracted out, a real, structure-specific
transfer effect remains, and it *is* larger for the isomorphic target than
the non-isomorphic control** — +0.131 vs. +0.071 in the early window
(roughly halved, and less reliable: 10/10 vs. 8/10 wins), +0.016 vs.
+0.010 overall (both small, isomorphic more reliable: 9/10 vs. 6/10
wins). This is the falsification criterion's control working as intended:
the effect shrinks and gets less consistent under the non-isomorphic
swap, but does not fully vanish to zero or below.

**Verdict against the stated falsification criterion**
(`research-agenda.md` #7: *"the advantage disappears once the domains are
made non-isomorphic"*): **not met in the strict sense** — some
structure-specific advantage survives the non-isomorphic control (+0.071
early, 8/10 wins) — but the honest reading is *weak, partial support*,
not a clean pass. Two things temper it further:

- The residual non-isomorphic structure-specific effect (+0.071, 8/10
  wins) could itself be partly explained by domain A's trained thresholds
  landing, by chance, in a *generically* useful region of the shared
  `[0,1]` raw-value range — not necessarily "structure" in the sense
  intended — and the sample size (10 trials) can't rule that out cleanly.
- By the `overall` metric (full 3000-step run), essentially everything
  washes out to small margins (0.010-0.026) with weaker win rates. **The
  transfer effect, such as it is, is a sample-efficiency / early-adaptation
  phenomenon, not a persistent asymptotic advantage** — which is itself
  worth stating precisely, since "transfer helps" without that qualifier
  overclaims what was actually measured.

## What this means for the master context's central "Trans" question

Section 17 of the master context asks: *"Can an intelligent system discover
reusable abstractions in one domain and apply them to structurally
different domains?"* This experiment's answer, at the scale tested: **a
little, and mostly as a head start, but a large part of what looks like
transfer in a naive comparison is actually just "not starting from
scratch," and that confound must be controlled for or the claim is not
credible.** This is exactly the kind of result `research-agenda.md` §21
("prefer controlled experiments over impressive demos") and §29
("identify what would make a serious researcher skeptical") exist to
surface — a skeptical reviewer's first question here would be "did you
control for the cold-start effect?", and this experiment did, and the
answer weakens the headline number by roughly half.

## What this does not establish

- **Small scale**: 4 frames, 1 learned-parameter class (`LearnedEmbeddingAgent`
  slots), 10 trials. Not swept across noise or switch frequency the way
  Experiments 1-2 were.
- **Only one direction of transfer tested** (finance-shaped → knowledge-shaped).
  Whether the effect is symmetric wasn't checked.
- **No test of transferring the belief-update *mechanism* or
  hyperparameters** (`switch_prob`, `lr`, `error_rate`) — only the learned
  `(w, b)` slot values. Those hyperparameters were identical across domains
  here by construction, so this experiment couldn't have detected a
  hyperparameter-transfer effect either way.
- **`random_differentiated`'s scale (3.0) was chosen by inspecting a few
  trained-expert magnitudes**, not tuned independently — a more rigorous
  version would sweep this scale to confirm the confound isn't sensitive
  to that choice.
