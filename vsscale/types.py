from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import vapoursynth as vs
from vskernels import Matrix

__all__ = [
    'TransferCurve', 'MatrixCoefficients'
]


class TransferCurve(IntEnum):
    IEC_61966 = vs.TRANSFER_IEC_61966_2_1
    BT709 = vs.TRANSFER_BT709
    BT601 = vs.TRANSFER_BT601
    ST240M = vs.TRANSFER_ST240_M
    BT2020_10bits = vs.TRANSFER_BT2020_10
    BT2020_12bits = vs.TRANSFER_BT2020_12

    LINEAR = vs.TRANSFER_LINEAR

    @classmethod
    def from_matrix(cls, matrix: Matrix) -> TransferCurve:
        if matrix not in _matrix_gamma_map:
            raise KeyError(
                'TransferCurve.from_matrix: curve is not supported!'
            )

        return _matrix_gamma_map[matrix]


class MatrixCoefficients(NamedTuple):
    k0: float
    phi: float
    alpha: float
    gamma: float

    @classmethod
    @property
    def SRGB(cls) -> MatrixCoefficients:
        return MatrixCoefficients(0.04045, 12.92, 0.055, 2.4)

    @classmethod
    @property
    def BT709(cls) -> MatrixCoefficients:
        return MatrixCoefficients(0.08145, 4.5, 0.0993, 2.22222)

    @classmethod
    @property
    def SMPTE240M(cls) -> MatrixCoefficients:
        return MatrixCoefficients(0.0912, 4.0, 0.1115, 2.22222)

    @classmethod
    @property
    def BT2020(cls) -> MatrixCoefficients:
        return MatrixCoefficients(0.08145, 4.5, 0.0993, 2.22222)

    @classmethod
    def from_curve(cls, curve: TransferCurve) -> MatrixCoefficients:
        if curve not in _gamma_linear_map:
            raise KeyError(
                'MatrixCoefficients.from_curve: curve is not supported!'
            )

        return _gamma_linear_map[curve]  # type: ignore


_gamma_linear_map = {
    TransferCurve.IEC_61966: MatrixCoefficients.SRGB,
    TransferCurve.BT709: MatrixCoefficients.BT709,
    TransferCurve.BT601: MatrixCoefficients.BT709,
    TransferCurve.ST240M: MatrixCoefficients.SMPTE240M,
    TransferCurve.BT2020_10bits: MatrixCoefficients.BT2020,
    TransferCurve.BT2020_12bits: MatrixCoefficients.BT2020
}

_matrix_gamma_map = {
    Matrix.BT709: TransferCurve.BT709,
    Matrix.BT470BG: TransferCurve.BT601,
    Matrix.SMPTE170M: TransferCurve.BT601,
    Matrix.SMPTE240M: TransferCurve.ST240M,
    Matrix.CHROMA_DERIVED_C: TransferCurve.IEC_61966,
    Matrix.ICTCP: TransferCurve.BT2020_10bits
}
