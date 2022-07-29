from __future__ import annotations

from typing import Any, Callable, NamedTuple, Protocol, Tuple, Union

import vapoursynth as vs
from vsexprtools import expr_func
from vskernels import Catrom, Kernel, VideoProp
from vskernels.kernels.abstract import Scaler
from vsmask.edge import EdgeDetect

__all__ = [
    'GenericScaler',
    'CreditMaskT', 'Resolution', 'DescaleAttempt'
]


CreditMaskT = Union[vs.VideoNode, Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode], EdgeDetect]


class _GeneriScaleNoShift(Protocol):
    def __call__(self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwds: Any) -> vs.VideoNode:
        ...


class _GeneriScaleWithShift(Protocol):
    def __call__(
        self, clip: vs.VideoNode, width: int, height: int, shift: Tuple[float, float],
        *args: Any, **kwds: Any
    ) -> vs.VideoNode:
        ...


class GenericScaler(Scaler):
    kernel: Kernel = Catrom()

    def __init__(
        self, func: _GeneriScaleNoShift | _GeneriScaleWithShift | Callable[..., vs.VideoNode], **kwargs: Any
    ) -> None:
        self.func = func
        self.kwargs = kwargs

    def scale(self, clip: vs.VideoNode, width: int, height: int, shift: Tuple[float, float] = (0, 0)) -> vs.VideoNode:
        if shift != (0, 0):
            try:
                return self.func(clip, width, height, shift, **self.kwargs)
            except BaseException:
                try:
                    return self.func(clip, width=width, height=height, shift=shift, **self.kwargs)
                except BaseException:
                    pass

        try:
            scaled = self.func(clip, width, height, **self.kwargs)
        except BaseException:
            scaled = self.func(clip, width=width, height=height, **self.kwargs)

        return self.kernel.shift(scaled, shift)


class Resolution(NamedTuple):
    """Tuple representing a resolution."""

    width: int

    height: int


class DescaleAttempt(NamedTuple):
    """Tuple representing a descale attempt."""

    """The native resolution."""
    resolution: Resolution

    """Descaled frame in native resolution."""
    descaled: vs.VideoNode

    """Descaled frame reupscaled with the same kernel."""
    rescaled: vs.VideoNode

    """The subtractive difference between the original and descaled frame."""
    diff: vs.VideoNode

    @classmethod
    def from_args(
        cls, clip: vs.VideoNode, width: int, height: int, shift: Tuple[float, float] = (0, 0),
        kernel: Kernel = Catrom(), **kwargs: VideoProp
    ) -> DescaleAttempt:
        descaled = kernel.descale(clip, width, height, shift)
        descaled = descaled.std.SetFrameProps(**kwargs)

        rescaled = kernel.scale(descaled, clip.width, clip.height)

        diff = expr_func([rescaled, clip], 'x y - abs').std.PlaneStats()

        resolution = Resolution(width, height)

        return DescaleAttempt(resolution, descaled, rescaled, diff)
