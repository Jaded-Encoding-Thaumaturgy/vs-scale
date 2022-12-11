from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Union

from vsexprtools import expr_func
from vskernels import Kernel
from vsmask.edge import EdgeDetect
from vstools import (
    ComparatorFunc, CustomIntEnum, CustomNotImplementedError, CustomStrEnum, Resolution, VSMapValue, merge_clip_props,
    vs
)

__all__ = [
    'CreditMaskT', 'Resolution', 'DescaleAttempt',
    'DescaleMode', 'DescaleResult', 'PlaneStatsKind',
    '_DescaleTypeGuards'
]


CreditMaskT = Union[vs.VideoNode, Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode], EdgeDetect]
"""@@PLACEHOLDER@@"""


class DescaleAttempt(NamedTuple):
    """Tuple representing a descale attempt."""

    resolution: Resolution
    """The native resolution."""

    descaled: vs.VideoNode
    """Descaled frame in native resolution."""

    rescaled: vs.VideoNode
    """Descaled frame reupscaled with the same kernel."""

    diff: vs.VideoNode
    """The subtractive difference between the original and descaled frame."""

    kernel: Kernel
    """Kernel used"""

    da_hash: str
    """Hash to identify the descale attempt"""

    @classmethod
    def get_hash(cls, width: int, height: int, kernel: Kernel) -> str:
        """@@PLACEHOLDER@@"""
        return f'{width}_{height}_{kernel.__class__.__name__}'

    @classmethod
    def from_args(
        cls, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float],
        kernel: Kernel, mode: DescaleMode, **kwargs: VSMapValue
    ) -> DescaleAttempt:
        """@@PLACEHOLDER@@"""

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
    """@@PLACEHOLDER@@"""

    descaled: vs.VideoNode
    """Descaled clip, can be var res"""

    rescaled: vs.VideoNode
    """Rescaled clip, can be var res"""

    upscaled: vs.VideoNode | None
    """Upscaled clip"""

    mask: vs.VideoNode | None
    """Descale error mask"""

    attempts: list[DescaleAttempt]
    """Descale attempts used"""

    out: vs.VideoNode
    """Normal output"""


class PlaneStatsKind(CustomStrEnum):
    """Type of PlaneStats comparing to use."""

    AVG = 'Average'
    MIN = 'Min'
    MAX = 'Max'
    DIFF = 'Diff'


@dataclass
class DescaleModeMeta:
    """@@PLACEHOLDER@@"""

    thr: float = field(default=5e-8)
    """@@PLACEHOLDER@@"""

    op: ComparatorFunc = field(default_factory=lambda: max)
    """@@PLACEHOLDER@@"""


class DescaleMode(DescaleModeMeta, CustomIntEnum):
    """@@PLACEHOLDER@@"""

    PlaneDiff = 0
    """@@PLACEHOLDER@@"""

    PlaneDiffMax = 1
    """@@PLACEHOLDER@@"""

    PlaneDiffMin = 2
    """@@PLACEHOLDER@@"""

    KernelDiff = 3
    """@@PLACEHOLDER@@"""

    KernelDiffMax = 4
    """@@PLACEHOLDER@@"""

    KernelDiffMin = 5
    """@@PLACEHOLDER@@"""

    def __call__(self, thr: float = 5e-8) -> DescaleMode:
        self.thr = thr  # TODO FIX THIS BECAUSE IT'S A FREAKIN' BUG!!!!!!!!!!!!!!!!

        return self

    @property
    def prop_key(self) -> str:
        """@@PLACEHOLDER@@"""

        if self.is_average:
            return 'PlaneStatsPAvg'
        elif self.is_kernel_diff:
            return 'PlaneStatsKDiff'

        raise CustomNotImplementedError

    @property
    def res_op(self) -> ComparatorFunc:
        """@@PLACEHOLDER@@"""

        if self in {self.PlaneDiff, self.KernelDiff, self.PlaneDiffMax, self.KernelDiffMax}:
            return max

        if self in {self.PlaneDiffMin, self.KernelDiffMin}:
            return min

        raise CustomNotImplementedError

    @property
    def diff_op(self) -> ComparatorFunc:
        """@@PLACEHOLDER@@"""

        if self in {self.PlaneDiff, self.KernelDiff, self.PlaneDiffMin, self.KernelDiffMin}:
            return min

        if self in {self.KernelDiffMax, self.PlaneDiffMax}:
            return max

        raise CustomNotImplementedError

    @property
    def is_average(self) -> bool:
        """@@PLACEHOLDER@@"""

        return self in {self.PlaneDiff, self.PlaneDiffMin, self.PlaneDiffMax}

    @property
    def is_kernel_diff(self) -> bool:
        """@@PLACEHOLDER@@"""

        return self in {self.KernelDiff, self.KernelDiffMin, self.KernelDiffMax}

    def prop_value(self, kind: PlaneStatsKind) -> str:
        """@@PLACEHOLDER@@"""

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
