from __future__ import annotations
from dataclasses import dataclass, field

from enum import IntEnum
from typing import Any, Callable, Iterable, NamedTuple, Protocol, Tuple, TypeVar, Union, overload

import vapoursynth as vs
from vsexprtools import expr_func
from vsexprtools.types import SupportsRichComparison, SupportsRichComparisonT
from vskernels import Catrom, Kernel, VideoProp
from vskernels.kernels.abstract import Scaler
from vsmask.edge import EdgeDetect

__all__ = [
    'GenericScaler',
    'CreditMaskT', 'Resolution', 'DescaleAttempt',
    '_ComparatorFunc', 'DescaleMode'
]

_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')


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

    """Hash to identify the descale attempt"""
    da_hash: str

    @classmethod
    def get_hash(cls, width: int, height: int, kernel: Kernel) -> str:
        return f'{width}_{height}_{kernel.__class__.__name__}'

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

        return DescaleAttempt(
            resolution, descaled, rescaled, diff, cls.get_hash(width, height, kernel)
        )


class _ComparatorFunc(Protocol):
    @overload
    def __call__(
        self, __arg1: SupportsRichComparisonT, __arg2: SupportsRichComparisonT,
        *_args: SupportsRichComparisonT, key: None = ...
    ) -> SupportsRichComparisonT:
        ...

    @overload
    def __call__(self, __arg1: _T, __arg2: _T, *_args: _T, key: Callable[[_T], SupportsRichComparison]) -> _T:
        ...

    @overload
    def __call__(self, __iterable: Iterable[SupportsRichComparisonT], *, key: None = ...) -> SupportsRichComparisonT:
        ...

    @overload
    def __call__(self, __iterable: Iterable[_T], *, key: Callable[[_T], SupportsRichComparison]) -> _T:
        ...

    @overload
    def __call__(
        self, __iterable: Iterable[SupportsRichComparisonT], *, key: None = ..., default: _T
    ) -> SupportsRichComparisonT | _T:
        ...

    @overload
    def __call__(
        self, __iterable: Iterable[_T1], *, key: Callable[[_T1], SupportsRichComparison], default: _T2
    ) -> _T1 | _T2:
        ...


@dataclass
class DescaleModeMeta:
    thr: float = field(default=0.0)
    op: _ComparatorFunc = field(default_factory=lambda: max)


class DescaleMode(DescaleModeMeta, IntEnum):
    PlaneAverage = 0
    PlaneDiff = 1

    def __call__(self, thr: float, op: _ComparatorFunc = max) -> DescaleMode:
        self.thr = thr
        self.op = op

        return self
