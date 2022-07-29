from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, NamedTuple, Protocol, Tuple, Union

import vapoursynth as vs
from vsexprtools import expr_func, ComparatorFunc
from vskernels import Catrom, Kernel, VideoProp
from vskernels.kernels.abstract import Scaler
from vsmask.edge import EdgeDetect

from .utils import merge_clip_props

__all__ = [
    'GenericScaler',
    'CreditMaskT', 'Resolution', 'DescaleAttempt',
    'DescaleMode', 'PlaneStatsKind'
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

    """Kernel used"""
    kernel: Kernel

    """Hash to identify the descale attempt"""
    da_hash: str

    @classmethod
    def get_hash(cls, width: int, height: int, kernel: Kernel) -> str:
        return f'{width}_{height}_{kernel.__class__.__name__}'

    @classmethod
    def from_args(
        cls, clip: vs.VideoNode, width: int, height: int, shift: Tuple[float, float],
        kernel: Kernel, mode: DescaleMode, **kwargs: VideoProp
    ) -> DescaleAttempt:
        descaled = kernel.descale(clip, width, height, shift)
        descaled = descaled.std.SetFrameProps(**kwargs)

        rescaled = kernel.scale(descaled, clip.width, clip.height)

        diff = expr_func([rescaled, clip], 'x y - abs').std.PlaneStats(
            None, prop=DescaleMode.PlaneAverage.prop_key
        )

        if mode in {DescaleMode.KernelDiff, DescaleMode.KernelDiffMin, DescaleMode.KernelDiffMax}:
            diff_props = rescaled.std.PlaneStats(
                clip, prop=DescaleMode.KernelDiff.prop_key
            )

            diff = merge_clip_props(diff, diff_props)

        resolution = Resolution(width, height)

        return DescaleAttempt(
            resolution, descaled, rescaled, diff, kernel, cls.get_hash(width, height, kernel)
        )


class PlaneStatsKind(str, Enum):
    AVG = 'Average'
    MIN = 'Min'
    MAX = 'Max'
    DIFF = 'Diff'


@dataclass
class DescaleModeMeta:
    thr: float = field(default=5e-8)
    op: ComparatorFunc = field(default_factory=lambda: max)


class DescaleMode(DescaleModeMeta, IntEnum):
    PlaneAverage = 0
    PlaneAverageMax = 1
    PlaneAverageMin = 2
    KernelDiff = 3
    KernelDiffMax = 4
    KernelDiffMin = 5

    def __call__(self, thr: float = 5e-8) -> DescaleMode:
        self.thr = thr

        return self

    @property
    def prop_key(self) -> str:
        if self.is_average:
            return 'PlaneStatsPAvg'
        elif self.is_kernel_diff:
            return 'PlaneStatsKDiff'

        raise RuntimeError

    @property
    def res_op(self) -> ComparatorFunc:
        if self in {self.PlaneAverage, self.KernelDiff, self.PlaneAverageMax, self.KernelDiffMax}:
            return max

        if self in {self.PlaneAverageMin, self.KernelDiffMin}:
            return min

        raise RuntimeError

    @property
    def diff_op(self) -> ComparatorFunc:
        if self in {self.PlaneAverage, self.KernelDiff, self.PlaneAverageMin, self.KernelDiffMin}:
            return min

        if self in {self.KernelDiffMax, self.PlaneAverageMax}:
            return max

        raise RuntimeError

    @property
    def is_average(self) -> bool:
        return self in {self.PlaneAverage, self.PlaneAverageMin, self.PlaneAverageMax}

    @property
    def is_kernel_diff(self) -> bool:
        return self in {self.KernelDiff, self.KernelDiffMin, self.KernelDiffMax}

    def prop_value(self, kind: PlaneStatsKind) -> str:
        return f'{self.prop_key}{kind.value}'

    def __hash__(self) -> int:
        return hash(self._name_)
