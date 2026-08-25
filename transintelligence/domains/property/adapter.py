from __future__ import annotations
from transintelligence import Context, Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner, ReasoningResult

def relative_value() -> ReasoningResult:
    ctx = Context(domain="property", objective="local market value comparison")
    property_entity = Entity("property", "Example Property")
    observations = [Observation(property_entity.id, "value_per_sqft", 610.0, "demo", context=ctx)]
    frame = ReferenceFrame("Local market", baseline=575.0, domain="property", metadata={"property":"value_per_sqft"})
    return BaselineRelativeReasoner(observations).evaluate(property_entity, frame)
