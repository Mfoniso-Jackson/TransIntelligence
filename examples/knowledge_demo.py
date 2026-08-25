from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transintelligence import Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner

def build_demo():
    a = Entity("idea", "Reference-frame-aware reasoning"); b = Entity("idea", "Universal scalar evaluation")
    observations = [Observation(a.id, "novelty", 0.82, "demo"), Observation(b.id, "novelty", 0.35, "demo")]
    frame = ReferenceFrame("Intellectual contribution", baseline=0.50, domain="knowledge", metadata={"property":"novelty"})
    reasoner = BaselineRelativeReasoner(observations)
    return reasoner.compare(a, b, frame)
if __name__ == "__main__": print(build_demo())
