from __future__ import annotations

from typing import Any

from vsexprtools import ExprOp, combine
from vskernels import Scaler
from vstools import EXPR_VARS, CustomIndexError, vs

from .helpers import GenericScaler

__all__ = [
    'MergeScalers'
]


class MergeScalers(GenericScaler):
    def __init__(self, *scalers: tuple[type[Scaler] | Scaler, float]) -> None:
        if (l := len(scalers)) < 2:
            raise CustomIndexError(f'Not enough scalers passed! ({l})', self.__class__)
        elif len(scalers) > len(EXPR_VARS):
            raise CustomIndexError(f'Too many scalers passed! ({l})')

        self.scalers = scalers

    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        scalers, weights = zip(*self.scalers)

        return combine(
            [scaler.scale(clip, width, height, shift, **kwargs) for scaler in scalers],
            ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV]
        )
