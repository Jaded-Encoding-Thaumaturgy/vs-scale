from __future__ import annotations

from typing import Callable, NamedTuple

import vapoursynth as vs

__all__ = [
    'CreditMaskT', 'Resolution', 'ScaleAttempt'
]


CreditMaskT = Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode]


class Resolution(NamedTuple):
    """Tuple representing a resolution."""

    width: int

    height: int


class ScaleAttempt(NamedTuple):
    """Tuple representing a descale attempt."""

    """The native resolution."""
    resolution: Resolution

    """Descaled frame in native resolution."""
    descaled: vs.VideoNode

    """Descaled frame reupscaled with the same kernel."""
    rescaled: vs.VideoNode

    """The subtractive difference between the original and descaled frame."""
    diff: vs.VideoNode
