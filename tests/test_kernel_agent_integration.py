"""Tests for KernelAgent's integration with the storage and event
primitives added after it (transintelligence/agents/base.py) -- the
first place SQLiteMemoryStore and EventStore are wired into an actual
agent loop, not just tested in isolation.
"""
from transintelligence.agents import KernelAgent
from transintelligence.core.events import EventStore
from transintelligence.memory.base import InMemoryStore
from transintelligence.storage import SQLiteMemoryStore


def test_default_construction_is_unchanged():
    """No events store, default memory -- the pre-existing behavior
    (tests/test_kernel.py) must keep working exactly as before."""
    agent = KernelAgent()
    step = agent.run_once({"event": "observe"})
    assert step.outcome["stored"] is True
    assert agent.memory.retrieve("agent_step")[0].content == step


def test_sqlite_memory_store_works_as_a_drop_in_replacement():
    """SQLiteMemoryStore structurally satisfies the same shape
    InMemoryStore does -- KernelAgent shouldn't need any special-casing
    to accept one instead."""
    sqlite_memory = SQLiteMemoryStore()
    agent = KernelAgent(memory=sqlite_memory)
    step = agent.run_once({"event": "observe"})
    stored = sqlite_memory.retrieve("agent_step")
    assert len(stored) == 1
    assert stored[0].content.outcome == step.outcome


def test_run_once_records_a_step_recorded_event_when_an_event_store_is_given():
    events = EventStore()
    agent = KernelAgent(events=events)
    agent.run_once({"event": "observe"})
    recorded = events.sequence_for("kernel_agent")
    assert len(recorded) == 1
    assert recorded[0].event_type == "step_recorded"


def test_no_events_are_recorded_when_no_event_store_is_given():
    agent = KernelAgent()
    agent.run_once({"event": "observe"})
    assert agent.events is None


def test_custom_agent_id_is_used_as_the_events_subject():
    events = EventStore()
    agent = KernelAgent(events=events, agent_id="agent_42")
    agent.run_once({"event": "observe"})
    assert events.sequence_for("agent_42")
    assert events.sequence_for("kernel_agent") == []


def test_memory_and_events_stay_in_sync_across_multiple_steps():
    """A real integration check: after N run_once() calls, both the
    memory store and the event log should reflect exactly N records,
    together, not just each in isolation."""
    memory = SQLiteMemoryStore()
    events = EventStore()
    agent = KernelAgent(memory=memory, events=events)
    for i in range(5):
        agent.run_once({"step": i})
    assert len(memory.retrieve("agent_step")) == 5
    assert len(events.sequence_for("kernel_agent")) == 5
