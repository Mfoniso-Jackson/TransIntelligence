# Experiment 8 results — causal discovery (PC-style skeleton + collider orientation)

Ran: `PYTHONPATH=. python experiments/exp08_causal_discovery/run.py`.
20 seeds per sample size, sample sizes N ∈ {100, 300, 1000, 3000}. See
`docs/research-agenda.md` #7e for the hypothesis and `docs/related-work.md`
§3c for the grounding theory (Spirtes & Glymour 1991).

Three structures, three separate questions — see the module docstring in
`experiments/exp08_causal_discovery/run.py` for why the confounding graph
here uses `TRUE_EFFECT_XY = 0.5` instead of experiment 6's `0.0`.

## 1. Skeleton recovery on the (reused-shape) confounding graph

Ground truth: 5 edges (`Z-X, Z-Y, X-Y, X-W, Y-W`). `W` is a **shielded**
collider — `X-Y` is also a direct edge — so this graph is a negative case
for orientation, not a positive one.

| n | mean precision | mean recall | W falsely oriented |
|---|---|---|---|
| 100 | 1.000 | 0.750 | 0/20 |
| 300 | 0.992 | 0.980 | 0/20 |
| 1000 | 0.992 | 1.000 | 0/20 |
| 3000 | 1.000 | 1.000 | 0/20 |

**Precision is at or near 1.000 at every sample size — the algorithm
essentially never proposes a spurious edge here, even at n=100 where it
still misses real ones.** Recall starts at 0.750 (some of the weaker,
noise-diluted edges like `X-W`/`Y-W` aren't statistically distinguishable
from independence yet at n=100) and reaches 1.000 by n=1000. **`W` is
never falsely oriented as a collider, at any sample size, across all 80
trials** — confirming the shielded-triple exclusion in `orient_colliders`
behaves exactly as designed, not just in the noiseless hand-check from
`tests/test_causal_reasoning.py`.

## 2. Collider orientation on a dedicated unshielded graph (A→B←C)

Ground truth: 2 edges (`A-B, B-C`), no `A-C` edge — the positive case for
v-structure orientation that the confounding graph's shielded `W` cannot
provide.

| n | mean precision | mean recall | correctly oriented |
|---|---|---|---|
| 100 | 1.000 | 1.000 | 20/20 |
| 300 | 1.000 | 1.000 | 20/20 |
| 1000 | 1.000 | 1.000 | 20/20 |
| 3000 | 1.000 | 1.000 | 20/20 |

**Skeleton recovery is exact and collider orientation is exact at every
sample size tested, including the smallest (n=100).** This is the
positive result: given a genuinely unshielded collider, discovery finds
it reliably, not just asymptotically — the effect sizes here (0.7/0.7
with 0.3 noise) are strong enough that this isn't a borderline case, but
it is a clean confirmation that the mechanism itself (skeleton via
Fisher-z partial correlation, then collider orientation via separating-set
membership) works exactly as the textbook algorithm predicts.

## 3. Negative control: four mutually independent variables

Ground truth: 0 edges. This is the confound control that could have
killed results 1 and 2 — if the false-edge rate had *grown* with N, it
would mean the algorithm just finds more "structure" wherever it looks
given enough statistical power, and the clean numbers above would be
worthless.

| n | trials with ≥1 false edge | mean false edges / trial |
|---|---|---|
| 100 | 2/20 | 0.100 |
| 300 | 2/20 | 0.100 |
| 1000 | 1/20 | 0.050 |
| 3000 | 0/20 | 0.000 |

**The false-edge rate does not grow with N — if anything it shrinks.**
At `alpha=0.01` with 6 candidate pairs tested per trial (4 variables, all
independent), a small residual false-positive rate is expected from
ordinary Type-I error across repeated independence tests; that it *stays
low and trends toward zero* rather than climbing confirms the discovery
procedure isn't an artifact of statistical power alone — more data makes
it more decisive about the *absence* of edges too, not just more willing
to propose edges.

## What this establishes about the causal-discovery mechanism

`discover_skeleton`/`orient_colliders` were verified against three
noiseless-mechanism hand-checks in `tests/test_causal_reasoning.py`
*before* this experiment was built: a chain (correct skeleton, correctly
left unoriented), an unshielded collider (correct skeleton, correctly
oriented), and a shielded triple built to mirror experiment 6's own graph
shape (correctly left unoriented despite `W` genuinely being a collider).
This experiment confirms the same three qualitative behaviors survive
realistic sampling noise across a 30x range of sample sizes, with a
dedicated negative control ruling out the obvious way the positive
results could have been an illusion of statistical power.

## What this does not establish

- **Meek's orientation-propagation rules are not implemented** (Meek,
  UAI 1995, pp. 403-410) — some edges that a full PC-algorithm
  implementation could additionally orient (via acyclicity and no-new-
  collider constraints, beyond direct v-structure detection) are left
  undirected here. This was a deliberate scope decision, not an
  oversight: skeleton discovery plus collider orientation is the smallest
  mechanism that can produce a falsifiable claim about structure
  discovery at all.
- **Linear-Gaussian data only** — the Fisher z-transform independence
  test assumes approximately linear relationships and Gaussian residuals,
  the same simplification `reasoning/causal/`'s effect-estimation and
  `reasoning/counterfactual/`'s abduction already make elsewhere in this
  codebase. Nonlinear dependencies that produce zero *linear* partial
  correlation (e.g. `Y = X^2 + noise` with `X` mean-zero) would be missed
  by this test entirely — untested here, and the natural next gap after
  nonlinear SCMs.
- **Small graphs only (3-4 nodes)** — `discover_skeleton`'s inner loop
  tests all subsets of the *combined* neighbor sets of x and y up to
  `len(nodes)-2` in size, not the sparser candidate-set pruning larger PC
  implementations use; fine at this repo's scale, not validated at scale.
- **No comparison against an alternative discovery algorithm** (e.g.
  score-based methods like GES) — only the constraint-based PC approach
  was implemented, so this experiment cannot say whether the specific
  errors observed (recall dip at low N) are characteristic of
  constraint-based discovery generally or specific to this
  implementation.
