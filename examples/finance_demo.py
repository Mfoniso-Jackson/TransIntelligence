from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transintelligence import Context, Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner

def build_demo():
    btc = Entity("asset", "BTC"); eth = Entity("asset", "ETH")
    ctx = Context(domain="finance", objective="risk-aware evaluation")
    observations = [Observation(btc.id, "volatility", 0.61, "demo", 0.94, context=ctx), Observation(eth.id, "volatility", 0.44, "demo", 0.93, context=ctx)]
    reasoner = BaselineRelativeReasoner(observations)
    historical = ReferenceFrame("Historical", baseline=0.50, domain="finance", assumptions=("demo volatility data",), metadata={"property":"volatility", "direction":"lower_is_better"})
    eth_relative = ReferenceFrame("Relative-to-ETH", baseline=0.44, domain="finance", assumptions=("ETH as comparator",), metadata={"property":"volatility", "direction":"lower_is_better"})
    portfolio = ReferenceFrame("Portfolio-risk", baseline=0.30, domain="finance", assumptions=("low risk mandate",), metadata={"property":"volatility", "direction":"lower_is_better"})
    return {"entities": (btc, eth), "evaluations": [reasoner.evaluate(btc, f) for f in (historical, eth_relative, portfolio)], "sensitivity": reasoner.sensitivity(btc, historical, portfolio)}
if __name__ == "__main__": print(build_demo())
