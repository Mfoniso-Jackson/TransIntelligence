from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Iterable

Vector = Iterable[float]

@dataclass(frozen=True)
class Trajectory:
    points: tuple[tuple[float, ...], ...]

class VectorReasoner:
    """Provider-agnostic vector baseline with no external model coupling."""
    def _v(self, x: Vector) -> tuple[float, ...]:
        return tuple(float(v) for v in x)
    def distance(self, x: Vector, y: Vector) -> float:
        xv, yv = self._v(x), self._v(y)
        if len(xv) != len(yv): raise ValueError("vectors must have the same dimensionality")
        return sqrt(sum((a - b) ** 2 for a, b in zip(xv, yv)))
    def similarity(self, x: Vector, y: Vector) -> float:
        xv, yv = self._v(x), self._v(y)
        if len(xv) != len(yv): raise ValueError("vectors must have the same dimensionality")
        denom = sqrt(sum(a*a for a in xv)) * sqrt(sum(b*b for b in yv))
        return 0.0 if denom == 0 else sum(a*b for a, b in zip(xv, yv)) / denom
    def nearest(self, x: Vector, items: list[Vector], k: int = 1) -> list[tuple[Vector, float]]:
        return sorted(((item, self.distance(x, item)) for item in items), key=lambda p: p[1])[:k]
