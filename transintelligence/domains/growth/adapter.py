from __future__ import annotations
from transintelligence import Context, Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner, ReasoningResult

def rank_prospects() -> ReasoningResult:
    ctx = Context(domain="growth", objective="acquisition prioritization")
    prospects = [Entity("prospect", "Community A"), Entity("prospect", "Business B")]
    observations = [Observation(prospects[0].id, "fit_score", 0.72, "demo", context=ctx), Observation(prospects[1].id, "fit_score", 0.58, "demo", context=ctx)]
    frame = ReferenceFrame("Acquisition fit", baseline=0.50, domain="growth", metadata={"property":"fit_score"})
    return BaselineRelativeReasoner(observations).rank(prospects, frame)
