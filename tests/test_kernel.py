from datetime import datetime, timezone, timedelta
import pytest
from transintelligence import Context, Entity, Observation, ReferenceFrame, Relationship, State, StateHistory
from transintelligence.reasoning.geometric import VectorReasoner
from transintelligence.reasoning.relative import BaselineRelativeReasoner
from transintelligence.epistemic import Claim, Confidence, Evidence, Source
from transintelligence.memory.base import InMemoryStore
from transintelligence.agents import KernelAgent

def test_entity_relationship_observation():
    e = Entity("asset", "BTC"); f = Entity("asset", "ETH")
    r = Relationship(e.id, f.id, "competes_with", directed=False)
    o = Observation(e.id, "volatility", 0.61, "unit", context=Context(domain="finance"), reference_frame=ReferenceFrame("market"))
    assert e.id and r.connects(f.id, e.id) and o.confidence == 1.0

def test_state_transitions_and_state_at():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc); s0 = State("env", {"regime":"calm"}, t0)
    s1 = s0.transition_to({"regime":"volatile"}, t0 + timedelta(days=1))
    h = StateHistory([s0, s1])
    assert h.state_at(t0 + timedelta(hours=12)).values["regime"] == "calm"
    assert h.trajectory(t0, t0 + timedelta(days=2)) == [s0, s1]

def test_reference_frame_relative_reasoning_sensitivity_and_invariance():
    x = Entity("asset", "BTC")
    obs = [Observation(x.id, "volatility", 0.61, "unit", 0.9), Observation(x.id, "liquidity", 0.80, "unit", 0.95)]
    reasoner = BaselineRelativeReasoner(obs)
    r1 = ReferenceFrame("historical", baseline=0.50, metadata={"property":"volatility", "direction":"lower_is_better"})
    r2 = ReferenceFrame("portfolio-risk", baseline=0.30, metadata={"property":"volatility", "direction":"lower_is_better"})
    assert reasoner.evaluate(x, r1).result != reasoner.evaluate(x, r2).result
    assert reasoner.sensitivity(x, r1, r2).result > 0
    a = ReferenceFrame("liq-a", baseline=0.70, metadata={"property":"liquidity"})
    b = ReferenceFrame("liq-b", baseline=0.70, observer="other", metadata={"property":"liquidity"})
    assert reasoner.evaluate(x, a).result == reasoner.evaluate(x, b).result

def test_relative_comparison_ranking_distance():
    a, b = Entity("idea", "A"), Entity("idea", "B")
    rr = BaselineRelativeReasoner([Observation(a.id, "score", 0.9, "unit"), Observation(b.id, "score", 0.4, "unit")])
    frame = ReferenceFrame("quality", baseline=0.0)
    assert rr.compare(a, b, frame).result > 0
    assert rr.rank([b, a], frame).result == [a, b]
    assert rr.relative_distance(a, b, frame).result == pytest.approx(0.5)

def test_geometric_reasoning():
    g = VectorReasoner()
    assert g.distance([0, 0], [3, 4]) == pytest.approx(5)
    assert g.similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert g.nearest([0, 0], [[2, 0], [1, 0]], 1)[0][0] == [1, 0]

def test_evidence_memory_agent_loop():
    ev = Evidence("measured volatility", Source("unit"), confidence=Confidence(0.9))
    claim = Claim("BTC volatility is elevated", evidence=(ev,), confidence=Confidence(0.8))
    mem = InMemoryStore(); mem.store(claim, tags=("claim",))
    step = KernelAgent(mem).run_once({"event":"observe"})
    assert claim.is_supported() and mem.retrieve("claim")[0].content == claim and step.outcome["stored"]
from transintelligence.core.observations import ObservationQuery, ObservationStore
from transintelligence.verification import EvidenceVerifier, VerificationStatus
from transintelligence.domains.growth import rank_prospects
from transintelligence.domains.property import relative_value

def test_observation_store_query_and_latest():
    ctx = Context(domain="finance")
    e = Entity("asset", "BTC")
    low = Observation(e.id, "volatility", 0.50, "unit", 0.5, context=ctx)
    high = Observation(e.id, "volatility", 0.61, "unit", 0.95, context=ctx)
    store = ObservationStore([low, high])
    assert store.query(ObservationQuery(context_domain="finance", min_confidence=0.9)) == [high]
    assert store.latest(e.id, "volatility") == high

def test_verifier_distinguishes_supported_and_unsupported_claims():
    verifier = EvidenceVerifier()
    unsupported = Claim("unsupported assertion")
    assert verifier.verify(unsupported).status == VerificationStatus.INSUFFICIENT_EVIDENCE
    ev = Evidence("measurement", Source("unit"), confidence=Confidence(0.8))
    supported = Claim("backed assertion", evidence=(ev,), confidence=Confidence(0.7))
    result = verifier.verify(supported)
    assert result.status == VerificationStatus.SUPPORTED and result.confidence == 0.7

def test_growth_and_property_adapters_use_core_reasoner():
    ranked = rank_prospects()
    valued = relative_value()
    assert [p.name for p in ranked.result] == ["Community A", "Business B"]
    assert valued.result > 0
