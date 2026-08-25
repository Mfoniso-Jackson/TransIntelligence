# Architecture

TransIntelligence is organized around a small kernel rather than product-specific applications. Core models live under `transintelligence.core`; representation-specific constructs such as reference frames live under `transintelligence.representation`; reasoning algorithms live behind protocols in `transintelligence.reasoning`.

Domain packages are lightweight adapter namespaces. They must depend on core primitives rather than embedding finance, property, growth, social, or knowledge logic inside the kernel.

Reasoning outputs are inspectable dataclasses that carry result, frame, assumptions, confidence, evidence, and explanation fields.
