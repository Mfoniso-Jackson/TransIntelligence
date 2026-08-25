# Agent Loop

The first agent abstraction is intentionally minimal and simulated:

Observe → Represent → Reason → Plan → Act → Observe outcome → Update memory.

`KernelAgent.run_once` records an observation-oriented step into memory. It does not connect to live external systems or perform consequential actions.
