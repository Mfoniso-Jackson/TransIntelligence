# Reasoning

`reasoning/interfaces.py` defines protocols for geometric, relative, temporal, causal, counterfactual, prediction, simulation, planning, and verification modules.

Implemented baselines:

- `VectorReasoner`: Euclidean distance, cosine similarity, and nearest neighbours over numeric vectors.
- `BaselineRelativeReasoner`: evaluates, compares, ranks, computes relative distance, and estimates frame sensitivity from observations and a reference frame.

These baselines are deterministic test scaffolds, not sophisticated AI models.


Verification is represented by `EvidenceVerifier`, which returns structured statuses for supported, contradicted, and insufficient-evidence claims.
