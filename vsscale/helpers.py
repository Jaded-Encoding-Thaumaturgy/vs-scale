from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Protocol

from vsaa import Nnedi3
from vskernels import Catrom, Kernel, KernelT, Scaler, ScalerT
from vstools import F_VD, MatrixT, get_w, plane, vs

from .types import Resolution

__all__ = [
    'GenericScaler',
    'scale_var_clip'
]


class _GeneriScaleNoShift(Protocol):
    def __call__(self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwds: Any) -> vs.VideoNode:
        ...


class _GeneriScaleWithShift(Protocol):
    def __call__(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float],
        *args: Any, **kwds: Any
    ) -> vs.VideoNode:
        ...


@dataclass
class GenericScaler(Scaler):
    kernel: KernelT = field(default_factory=lambda: Catrom, kw_only=True)
    scaler: ScalerT | None = field(default=None, kw_only=True)
    shifter: KernelT | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        self._kernel = Kernel.ensure_obj(self.kernel, self.__class__)
        self._scaler = self._kernel.ensure_obj(self.scaler, self.__class__)
        self._shifter = Kernel.ensure_obj(
            self.shifter or (self._scaler if isinstance(self._scaler, Kernel) else Catrom), self.__class__
        )

    def __init__(
        self, func: _GeneriScaleNoShift | _GeneriScaleWithShift | F_VD, **kwargs: Any
    ) -> None:
        self.func = func
        self.kwargs = kwargs

    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        kwargs = self.kwargs | kwargs

        output = None

        if shift != (0, 0):
            try:
                output = self.func(clip, width, height, shift, **kwargs)
            except BaseException:
                try:
                    output = self.func(clip, width=width, height=height, shift=shift, **kwargs)
                except BaseException:
                    pass

        if output is None:
            try:
                output = self.func(clip, width, height, **kwargs)
            except BaseException:
                output = self.func(clip, width=width, height=height, **kwargs)

        return self._finish_scale(output, clip, width, height, shift)

    def _finish_scale(
        self, clip: vs.VideoNode, input_clip: vs.VideoNode, width: int, height: int,
        shift: tuple[float, float] = (0, 0), matrix: MatrixT | None = None
    ) -> vs.VideoNode:
        assert input_clip.format
        if input_clip.format.num_planes == 1:
            clip = plane(clip, 0)

        if (clip.width, clip.height) != (width, height):
            clip = self._scaler.scale(clip, width, height)

        if shift != (0, 0):
            clip = self._shifter.shift(clip, shift)

        assert clip.format

        if clip.format.id == input_clip.format.id:
            return clip

        return self._kernel.resample(clip, input_clip, matrix)


def scale_var_clip(
    clip: vs.VideoNode,
    width: int | Callable[[Resolution], int] | None, height: int | Callable[[Resolution], int],
    shift: tuple[float, float] | Callable[[Resolution], tuple[float, float]] = (0, 0),
    scaler: Scaler | Callable[[Resolution], Scaler] = Nnedi3(), debug: bool = False
) -> vs.VideoNode:
    if not debug:
        try:
            return scaler.scale(clip, width, height, shift)  # type: ignore
        except BaseException:
            pass

    _cached_clips = dict[str, vs.VideoNode]()

    no_accepts_var = list[Scaler]()

    def _eval_scale(f: vs.VideoFrame, n: int) -> vs.VideoNode:
        key = f'{f.width}_{f.height}'

        if key not in _cached_clips:
            res = Resolution(f.width, f.height)

            norm_scaler = scaler(res) if callable(scaler) else scaler
            norm_shift = shift(res) if callable(shift) else shift
            norm_height = height(res) if callable(height) else height

            if width is None:
                norm_width = get_w(norm_height, res.width / res.height)
            else:
                norm_width = width(res) if callable(width) else width

            part_scaler = partial(
                norm_scaler.scale, width=norm_width, height=norm_height, shift=norm_shift
            )

            scaled = clip
            if (scaled.width, scaled.height) != (norm_width, norm_height):
                if norm_scaler not in no_accepts_var:
                    try:
                        scaled = part_scaler(clip)
                    except BaseException:
                        no_accepts_var.append(norm_scaler)

                if norm_scaler in no_accepts_var:
                    const_clip = clip.resize.Point(res.width, res.height)

                    scaled = part_scaler(const_clip)

            if debug:
                scaled = scaled.std.SetFrameProps(var_width=res.width, var_height=res.height)

            _cached_clips[key] = scaled

        return _cached_clips[key]

    if callable(width) or callable(height):
        out_clip = clip
    else:
        out_clip = clip.std.BlankClip(width, height)

    return out_clip.std.FrameEval(_eval_scale, clip, clip)
