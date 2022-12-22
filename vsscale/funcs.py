from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Concatenate, Literal, cast

from vsaa import Nnedi3
from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Scaler, ScalerT
from vsmasktools import ringing_mask
from vsrgtools import LimitFilterMode, RepairMode, limit_filter, median_clips, repair, unsharp_masked
from vstools import (
    EXPR_VARS, ColorRange, CustomIndexError, CustomOverflowError, P, check_ref_clip, inject_self, scale_8bit, vs
)

from .helpers import GenericScaler
from .shaders import FSRCNNXShader, FSRCNNXShaderT

__all__ = [
    'MergeScalers',
    'ClampScaler', 'MergedFSRCNNX',
    'UnsharpLimitScaler', 'UnsharpedFSRCNNX'
]


class MergeScalers(GenericScaler):
    def __init__(self, *scalers: ScalerT | tuple[ScalerT, float | None]) -> None:
        if (l := len(scalers)) < 2:
            raise CustomIndexError(f'Not enough scalers passed! ({l})', self.__class__)
        elif len(scalers) > len(EXPR_VARS):
            raise CustomIndexError(f'Too many scalers passed! ({l})', self.__class__)

        if any(isinstance(s, tuple) for s in scalers):
            norm_scalers = [
                scaler if isinstance(scaler, tuple) else (scaler, None) for scaler in scalers
            ]

            curr_sum = 0.0
            n_auto_weight = 0

            for i, (_, weight) in enumerate(norm_scalers):
                if weight is None:
                    n_auto_weight += 1
                elif weight <= 0.0:
                    raise CustomOverflowError(
                        f'Weights have to be positive, >= 0.0! (Scaler index: ({i})', self.__class__
                    )
                else:
                    curr_sum += weight

            if curr_sum >= 1.0:
                raise CustomOverflowError(
                    'Sum of the weights should be less than 1.0!', self.__class__
                )

            if n_auto_weight:
                a_wgh = (1.0 - curr_sum) / n_auto_weight

                norm_scalers = [
                    (scaler, a_wgh if weight is None else weight)
                    for scaler, weight in norm_scalers
                ]
        else:
            weight = 1.0 / len(scalers)

            norm_scalers = [(scaler, weight) for scaler in scalers]  # type: ignore

        self.scalers = [
            (
                Scaler.ensure_obj(scaler, self.__class__),
                float(weight if weight else 0)
            ) for scaler, weight in norm_scalers
        ]

    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        scalers, weights = cast(tuple[list[Scaler], list[float]], zip(*self.scalers))

        return combine(
            [scaler.scale(clip, width, height, shift, **kwargs) for scaler in scalers],
            ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV]
        )


@dataclass
class ClampScaler(GenericScaler):
    ref_scaler: ScalerT

    strength: int = 80

    overshoot: float | None = None
    undershoot: float | None = None

    limit: RepairMode | bool = True

    operator: Literal[ExprOp.MAX, ExprOp.MIN] | None = ExprOp.MIN
    masked: bool = True

    reference: ScalerT | vs.VideoNode = Nnedi3(0, opencl=None)

    range_out: ColorRange | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.strength >= 100:
            raise CustomOverflowError('strength can\'t be more or equal to 100!', self.__class__)
        elif self.strength <= 0:
            raise CustomOverflowError('strength can\'t be less or equal to 0!', self.__class__)

        if self.overshoot is None:
            self.overshoot = self.strength / 100
        if self.undershoot is None:
            self.undershoot = self.overshoot

        self._ref_scaler = Scaler.ensure_obj(self.ref_scaler, self.__class__)

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0),
        *, smooth: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        assert (self.undershoot or self.undershoot == 0) and (self.overshoot or self.overshoot == 0)

        fsrcnnx = self._ref_scaler.scale(clip, width, height, shift, **kwargs)

        if isinstance(self.reference, vs.VideoNode):
            smooth = self.reference  # type: ignore

            if shift != (0, 0):
                smooth = self._kernel.shift(smooth, shift)  # type: ignore
        else:
            smooth = Scaler.ensure_obj(self.reference, self.__class__).scale(clip, width, height, shift)  # type: ignore

        assert smooth

        check_ref_clip(fsrcnnx, smooth)

        range_out = ColorRange.from_video(clip, False) if self.range_out is None else self.range_out

        fsr_weight = self.strength / 100

        if self.limit is True:
            expression = [
                'x {fsr_weight} * y {ref_weight} * + up!',
                '{overshoot} O!', '{undershoot} U!',
                'up@ z O@ + > z O@ + up@ ? a U@ - < a U@ - up@ z O@ + > z O@ + up@ ? ?'
            ]

            if range_out is ColorRange.LIMITED:
                expression.append(f'{scale_8bit(clip, 16)} {{clamp_max}} clamp')

            merged = norm_expr(
                [fsrcnnx, smooth, smooth.std.Maximum(), smooth.std.Minimum()],
                expression, fsr_weight=fsr_weight, ref_weight=1.0 - fsr_weight,
                undershoot=self.undershoot * (2 ** 8), overshoot=self.overshoot * (2 ** 8),
                clamp_max=[scale_8bit(clip, 235), scale_8bit(clip, 240)]
            )
        else:
            merged = smooth.std.Merge(fsrcnnx, fsr_weight)

            if isinstance(self.limit, RepairMode):
                merged = repair(merged, smooth, self.limit)

        if self.operator is not None:
            merge2 = combine([smooth, fsrcnnx], ExprOp.MIN)

            if self.masked:
                merged = merged.std.MaskedMerge(merge2, ringing_mask(smooth))
            else:
                merged = merge2
        elif self.masked:
            merged.std.MaskedMerge(smooth, ringing_mask(smooth))

        return merged


class UnsharpLimitScaler(GenericScaler):
    def __init__(
        self, ref_scaler: ScalerT,
        unsharp_func: Callable[
            Concatenate[vs.VideoNode, P], vs.VideoNode
        ] = partial(unsharp_masked, radius=2, strength=65),
        merge_mode: LimitFilterMode | bool = True,
        reference: ScalerT | vs.VideoNode = Nnedi3(0, opencl=None),
        *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self.unsharp_func = unsharp_func

        self.merge_mode = merge_mode

        self.reference = reference
        self.ref_scaler = Scaler.ensure_obj(ref_scaler, self.__class__)

        self.args = args
        self.kwargs = kwargs

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0),
        *, smooth: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        fsrcnnx = self.ref_scaler.scale(clip, width, height, shift, **kwargs)

        if isinstance(self.reference, vs.VideoNode):
            smooth = self.reference  # type: ignore

            if shift != (0, 0):
                smooth = self._kernel.shift(smooth, shift)  # type: ignore
        else:
            smooth = Scaler.ensure_obj(self.reference, self.__class__).scale(clip, width, height, shift)  # type: ignore

        assert smooth

        check_ref_clip(fsrcnnx, smooth)

        smooth_sharp = self.unsharp_func(smooth, *self.args, **self.kwargs)

        if isinstance(self.merge_mode, LimitFilterMode):
            return limit_filter(smooth, fsrcnnx, smooth_sharp, self.merge_mode)

        if self.merge_mode:
            return median_clips(smooth, fsrcnnx, smooth_sharp)

        return combine([smooth, fsrcnnx, smooth_sharp], ExprOp.MIN)


@dataclass
class MergedFSRCNNX(ClampScaler):
    ref_scaler: FSRCNNXShaderT = field(default_factory=lambda: FSRCNNXShader.x56, kw_only=True)


class UnsharpedFSRCNNX(UnsharpLimitScaler):
    def __init__(
        self,
        unsharp_func: Callable[
            Concatenate[vs.VideoNode, P], vs.VideoNode
        ] = partial(unsharp_masked, radius=2, strength=65),
        merge_mode: LimitFilterMode | bool = True,
        reference: ScalerT | vs.VideoNode = Nnedi3(0, opencl=None),
        ref_scaler: ScalerT = FSRCNNXShader.x56,
        *args: P.args, **kwargs: P.kwargs
    ) -> None:
        super().__init__(ref_scaler, unsharp_func, merge_mode, reference, *args, **kwargs)
