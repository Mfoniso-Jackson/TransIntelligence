# Reference Frames

A reference frame makes contextual evaluation explicit: `evaluate(x, reference_frame)`. Frames may include baseline, observer, objective, time window, domain, scale, assumptions, constraints, context, and metadata.

Frames are composable using `compose` and comparable using `differences`. Baseline relative reasoning uses frame metadata such as `property` and `direction` to avoid silently assuming a universal metric.
