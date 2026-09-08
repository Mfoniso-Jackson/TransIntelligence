from __future__ import annotations
from dataclasses import dataclass, field
from transintelligence.core.common import Metadata, TimeWindow, new_id
from transintelligence.core.contexts import Context

@dataclass(frozen=True)
class ReferenceFrame:
    name: str
    baseline: float | str | None = None
    observer: str | None = None
    objective: str | None = None
    time_window: TimeWindow | None = None
    domain: str | None = None
    scale: str | None = None
    assumptions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    context: Context | None = None
    metadata: Metadata = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("rf"))

    def compose(self, other: "ReferenceFrame") -> "ReferenceFrame":
        return ReferenceFrame(
            name=f"{self.name}+{other.name}", baseline=other.baseline if other.baseline is not None else self.baseline,
            observer=other.observer or self.observer, objective=other.objective or self.objective,
            time_window=other.time_window or self.time_window, domain=other.domain or self.domain,
            scale=other.scale or self.scale, assumptions=(*self.assumptions, *other.assumptions),
            constraints=(*self.constraints, *other.constraints),
            context=self.context.compose(other.context) if self.context and other.context else other.context or self.context,
            metadata={**self.metadata, **other.metadata},
        )

    def differences(self, other: "ReferenceFrame") -> dict[str, tuple[object, object]]:
        fields = ("baseline", "observer", "objective", "domain", "scale", "assumptions", "constraints")
        return {f: (getattr(self, f), getattr(other, f)) for f in fields if getattr(self, f) != getattr(other, f)}

    def validate(self) -> tuple[str, ...]:
        """Structural sanity checks on this frame's own fields --
        returns a tuple of human-readable issue descriptions (empty if
        none), the same non-raising, traceable-output convention already
        used elsewhere in this codebase (`VerificationResult`,
        `CalibrationResult`) rather than an exception: a frame failing
        validation is informative diagnostic output a caller can act on,
        not necessarily a fatal error at construction time -- a
        `ReferenceFrame` is still a plain frozen dataclass, constructible
        with any field values, exactly as before."""
        issues: list[str] = []
        if not self.name:
            issues.append("name must be non-empty")
        if self.domain is not None and not self.domain.strip():
            issues.append("domain, if set, must be non-empty")
        if (self.time_window is not None and self.time_window.start is not None
                and self.time_window.end is not None and self.time_window.start > self.time_window.end):
            issues.append(f"time_window.start ({self.time_window.start}) is after time_window.end ({self.time_window.end})")
        return tuple(issues)
