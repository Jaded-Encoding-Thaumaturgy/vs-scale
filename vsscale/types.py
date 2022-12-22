from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Union

from vsexprtools import expr_func
from vskernels import Kernel
from vsmasktools import EdgeDetectT
from vstools import (
    ComparatorFunc, CustomIntEnum, CustomNotImplementedError, CustomStrEnum, Resolution, VSMapValue, merge_clip_props,
    vs
)

__all__ = [
    'CreditMaskT', 'Resolution', 'DescaleAttempt',
    'DescaleMode', 'DescaleResult', 'PlaneStatsKind',
    '_DescaleTypeGuards'
]


CreditMaskT = Union[vs.VideoNode, Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode], EdgeDetectT]


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
        cls, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float],
        kernel: Kernel, mode: DescaleMode, **kwargs: VSMapValue
    ) -> DescaleAttempt:
        descaled = kernel.descale(clip, width, height, shift)
        descaled = descaled.std.SetFrameProps(**kwargs)

        rescaled = kernel.scale(descaled, clip.width, clip.height)

        diff = expr_func([rescaled, clip], 'x y - abs').std.PlaneStats(
            None, prop=DescaleMode.PlaneDiff.prop_key
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


@dataclass
class DescaleResult:
    """Descaled clip, can be var res"""
    descaled: vs.VideoNode

    """Rescaled clip, can be var res"""
    rescaled: vs.VideoNode

    """Upscaled clip"""
    upscaled: vs.VideoNode | None

    """Descale error mask"""
    mask: vs.VideoNode | None

    """Descale attempts used"""
    attempts: list[DescaleAttempt]

    """Normal output"""
    out: vs.VideoNode


class PlaneStatsKind(CustomStrEnum):
    AVG = 'Average'
    MIN = 'Min'
    MAX = 'Max'
    DIFF = 'Diff'


@dataclass
class DescaleModeMeta:
    thr: float = field(default=5e-8)
    op: ComparatorFunc = field(default_factory=lambda: max)


class DescaleMode(DescaleModeMeta, CustomIntEnum):
    PlaneDiff = 0
    PlaneDiffMax = 1
    PlaneDiffMin = 2
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

        raise CustomNotImplementedError

    @property
    def res_op(self) -> ComparatorFunc:
        if self in {self.PlaneDiff, self.KernelDiff, self.PlaneDiffMax, self.KernelDiffMax}:
            return max

        if self in {self.PlaneDiffMin, self.KernelDiffMin}:
            return min

        raise CustomNotImplementedError

    @property
    def diff_op(self) -> ComparatorFunc:
        if self in {self.PlaneDiff, self.KernelDiff, self.PlaneDiffMin, self.KernelDiffMin}:
            return min

        if self in {self.KernelDiffMax, self.PlaneDiffMax}:
            return max

        raise CustomNotImplementedError

    @property
    def is_average(self) -> bool:
        return self in {self.PlaneDiff, self.PlaneDiffMin, self.PlaneDiffMax}

    @property
    def is_kernel_diff(self) -> bool:
        return self in {self.KernelDiff, self.KernelDiffMin, self.KernelDiffMax}

    def prop_value(self, kind: PlaneStatsKind) -> str:
        return f'{self.prop_key}{kind.value}'

    def __hash__(self) -> int:
        return hash(self._name_)


class _DescaleTypeGuards:
    class _UpscalerNotNone(DescaleResult):
        upscaled: vs.VideoNode

    class _UpscalerIsNone(DescaleResult):
        upscaled: None

    class _MaskNotNone(DescaleResult):
        mask: vs.VideoNode

    class _MaskIsNone(DescaleResult):
        mask: None

    class UpscalerNotNoneMaskNotNone(_UpscalerNotNone, _MaskNotNone, DescaleResult):
        ...

    class UpscalerNotNoneMaskIsNone(_UpscalerNotNone, _MaskIsNone, DescaleResult):
        ...

    class UpscalerIsNoneMaskNotNone(_UpscalerIsNone, _MaskNotNone, DescaleResult):
        ...

    class UpscalerIsNoneMaskIsNone(_UpscalerIsNone, _MaskIsNone, DescaleResult):
        ...
