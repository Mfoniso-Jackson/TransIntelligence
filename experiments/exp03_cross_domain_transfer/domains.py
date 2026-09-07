"""Domain definitions for Experiment 3 (docs/research-agenda.md #7).

Per Gentner's structure-mapping theory (1983, see
docs/related-work.md #6): two domains are *analogous* if they share a
system of interconnected *relations* while differing in surface
*attributes*; they are merely *similar* (or trivially relabeled) if they
share attributes instead. The mechanics of FrameSwitchEnv and
LearnedEmbeddingAgent only ever read a frame's numeric `baseline` and
`direction` -- never its `name` or the entity/property labels -- so the
"structure" being shared or broken here is exactly that numeric relational
system (which baselines, which directions, how many frames), and the
"attributes" being varied are exactly the human-facing labels (domain,
entity type, property name, frame names) that the mechanics never touch.

DOMAIN_A: finance-shaped (assets, volatility).
DOMAIN_B_ISOMORPHIC: knowledge-shaped (ideas, novelty), but with the
    identical (baseline, direction) numeric structure as DOMAIN_A -- same
    relations, different attributes. A strategy trained on DOMAIN_A's
    numeric structure is, by construction, also correct for this domain.
DOMAIN_B_NONISOMORPHIC: knowledge-shaped, same labels as
    DOMAIN_B_ISOMORPHIC, but with a genuinely different (baseline,
    direction) numeric structure -- the control. If "transfer" survives
    swapping the isomorphic domain for this one, it wasn't testing
    structure at all.
"""
from __future__ import annotations

from transintelligence import ReferenceFrame

DOMAIN_A_FRAMES = [
    ReferenceFrame("Historical", baseline=0.5, domain="finance", metadata={"direction": "higher_is_better"}),
    ReferenceFrame("Relative-to-ETH", baseline=0.3, domain="finance", metadata={"direction": "higher_is_better"}),
    ReferenceFrame("Portfolio-risk", baseline=0.5, domain="finance", metadata={"direction": "lower_is_better"}),
    ReferenceFrame("Market-regime", baseline=0.7, domain="finance", metadata={"direction": "lower_is_better"}),
]

# Same (baseline, direction) sequence as DOMAIN_A_FRAMES, different labels.
DOMAIN_B_ISOMORPHIC_FRAMES = [
    ReferenceFrame("Peer-review-baseline", baseline=0.5, domain="knowledge", metadata={"direction": "higher_is_better"}),
    ReferenceFrame("Field-average", baseline=0.3, domain="knowledge", metadata={"direction": "higher_is_better"}),
    ReferenceFrame("Citation-caution", baseline=0.5, domain="knowledge", metadata={"direction": "lower_is_better"}),
    ReferenceFrame("Hype-threshold", baseline=0.7, domain="knowledge", metadata={"direction": "lower_is_better"}),
]

# Same labels as the isomorphic version, deliberately different (baseline,
# direction) numeric structure -- not a permutation of DOMAIN_A's values,
# a genuinely different rule set.
DOMAIN_B_NONISOMORPHIC_FRAMES = [
    ReferenceFrame("Peer-review-baseline", baseline=0.2, domain="knowledge", metadata={"direction": "lower_is_better"}),
    ReferenceFrame("Field-average", baseline=0.6, domain="knowledge", metadata={"direction": "lower_is_better"}),
    ReferenceFrame("Citation-caution", baseline=0.4, domain="knowledge", metadata={"direction": "higher_is_better"}),
    ReferenceFrame("Hype-threshold", baseline=0.8, domain="knowledge", metadata={"direction": "higher_is_better"}),
]
