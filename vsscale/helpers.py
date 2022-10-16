from __future__ import annotations

from typing import Protocol, Any
from dataclasses import field, dataclass

from vstools import vs, F_VD
from vskernels import Catrom, Kernel, Scaler

__all__ = [
    'GenericScaler'
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
    kernel: type[Kernel] | Kernel = field(default_factory=lambda: Catrom(), kw_only=True)

    def __init__(
        self, func: _GeneriScaleNoShift | _GeneriScaleWithShift | F_VD, **kwargs: Any
    ) -> None:
        self.func = func
        self.kwargs = kwargs

    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        kwargs = self.kwargs | kwargs

        if shift != (0, 0):
            try:
                return self.func(clip, width, height, shift, **kwargs)
            except BaseException:
                try:
                    return self.func(clip, width=width, height=height, shift=shift, **kwargs)
                except BaseException:
                    pass

        try:
            scaled = self.func(clip, width, height, **kwargs)
        except BaseException:
            scaled = self.func(clip, width=width, height=height, **kwargs)

        return self.kernel.shift(scaled, shift)
