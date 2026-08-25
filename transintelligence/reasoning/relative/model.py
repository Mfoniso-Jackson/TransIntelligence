from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from transintelligence.core.observations import Observation, ObservationQuery, ObservationStore
from transintelligence.representation.reference_frames import ReferenceFrame

@dataclass(frozen=True)
class ReasoningResult:
    result: Any
    reference_frame: ReferenceFrame
    assumptions: tuple[str, ...] = ()
    confidence: float = 0.5
    supporting_observations: tuple[Observation, ...] = ()
    explanation: str = ""

class BaselineRelativeReasoner:
    def __init__(self, observations: list[Observation] | ObservationStore | None = None, property_name: str = "score"):
        self.observations = observations if isinstance(observations, ObservationStore) else ObservationStore(observations or [])
        self.property_name = property_name
    def _score(self, x: Any, frame: ReferenceFrame) -> tuple[float, tuple[Observation, ...], float]:
        entity_id = getattr(x, "id", str(x))
        prop = str(frame.metadata.get("property", self.property_name))
        obs = tuple(self.observations.query(ObservationQuery(entity_id=entity_id, property_name=prop)))
        if not obs: return (0.0, (), 0.1)
        raw = float(obs[-1].value)
        baseline = frame.baseline if isinstance(frame.baseline, (int, float)) else frame.metadata.get("baseline", 0.0)
        direction = frame.metadata.get("direction", "higher_is_better")
        score = raw - float(baseline)
        if direction == "lower_is_better": score = -score
        return score, obs, min(o.confidence for o in obs)
    def evaluate(self, x: Any, reference_frame: ReferenceFrame) -> ReasoningResult:
        score, obs, conf = self._score(x, reference_frame)
        return ReasoningResult(score, reference_frame, reference_frame.assumptions, conf, obs, "baseline-adjusted deterministic score")
    def compare(self, x: Any, y: Any, reference_frame: ReferenceFrame) -> ReasoningResult:
        ex, ey = self.evaluate(x, reference_frame), self.evaluate(y, reference_frame)
        return ReasoningResult(ex.result - ey.result, reference_frame, reference_frame.assumptions, min(ex.confidence, ey.confidence), ex.supporting_observations + ey.supporting_observations, "positive means x ranks above y in frame")
    def rank(self, items: list[Any], reference_frame: ReferenceFrame) -> ReasoningResult:
        ranked = sorted(items, key=lambda i: self.evaluate(i, reference_frame).result, reverse=True)
        obs = tuple(o for i in ranked for o in self.evaluate(i, reference_frame).supporting_observations)
        return ReasoningResult(ranked, reference_frame, reference_frame.assumptions, min([o.confidence for o in obs], default=0.1), obs)
    def relative_distance(self, x: Any, y: Any, reference_frame: ReferenceFrame) -> ReasoningResult:
        cmp = self.compare(x, y, reference_frame)
        return ReasoningResult(abs(cmp.result), reference_frame, cmp.assumptions, cmp.confidence, cmp.supporting_observations)
    def sensitivity(self, x: Any, r1: ReferenceFrame, r2: ReferenceFrame) -> ReasoningResult:
        e1, e2 = self.evaluate(x, r1), self.evaluate(x, r2)
        return ReasoningResult(abs(e1.result - e2.result), r2, (*r1.assumptions, *r2.assumptions), min(e1.confidence, e2.confidence), e1.supporting_observations + e2.supporting_observations, f"frame differences: {r1.differences(r2)}")
