from __future__ import annotations

from vsexprtools import ExprOp, aka_expr_available, norm_expr
from vstools import CustomValueError, MatrixCoefficients, Transfer, get_depth, vs

__all__ = [
    'gamma2linear', 'linear2gamma'
]


def _linear_diff(cont: float, thr: float, diff: str = '') -> str:
    inv_op = f'{diff} -' if diff else ''

    return f'1 1 {cont} {thr} {inv_op} * exp + /'


def _sigmoid_x(sigmoid: bool, cont: float, thr: float) -> tuple[str, str, str]:
    if not sigmoid:
        return '', '', ''

    header, x0, x1 = '', _linear_diff(cont, thr), _linear_diff(cont, thr, '1')

    if aka_expr_available:
        header = f'{x0} SX0! {x1} SX1!'
        x0, x1 = 'SX0@', 'SX1@'

    return header, x0, x1


def _clamp_converted(clip: vs.VideoNode, header: str, expr: str, curve: Transfer, sigmoid: bool) -> vs.VideoNode:
    linear = norm_expr(clip, f'{header} {expr} {ExprOp.clamp(0, 1)}', planes=0 if sigmoid else None)

    return linear.std.SetFrameProps(_Transfer=curve.value)


def gamma2linear(
    clip: vs.VideoNode, curve: Transfer, gcor: float = 1.0,
    sigmoid: bool = False, thr: float = 0.5, cont: float = 6.5,
    epsilon: float = 1e-6
) -> vs.VideoNode:
    """Convert gamma to linear curve."""
    assert clip.format

    if get_depth(clip) != 32 and clip.format.sample_type != vs.FLOAT:
        raise CustomValueError('Your clip must be 32bit float!', gamma2linear)

    c = MatrixCoefficients.from_transfer(curve)

    header, x0, x1 = _sigmoid_x(sigmoid, cont, thr)

    expr = f'x {c.k0} <= x {c.phi} / x {c.alpha} + 1 {c.alpha} + / {c.gamma} pow ? {gcor} pow'

    if sigmoid:
        expr = f'{thr} 1 {expr} {x1} {x0} - * {x0} + {epsilon} max / 1 - {epsilon} max log {cont} / -'

    return _clamp_converted(clip, header, expr, Transfer.LINEAR, sigmoid)


def linear2gamma(
    clip: vs.VideoNode, curve: Transfer, gcor: float = 1.0,
    sigmoid: bool = False, thr: float = 0.5, cont: float = 6.5
) -> vs.VideoNode:
    """Convert linear curve to gamma."""
    assert clip.format

    if get_depth(clip) != 32 and clip.format.sample_type != vs.FLOAT:
        raise CustomValueError('Your clip must be 32bit float!', linear2gamma)

    c = MatrixCoefficients.from_transfer(curve)

    header, x0, x1 = _sigmoid_x(sigmoid, cont, thr)

    if sigmoid:
        lin = f"{_linear_diff(cont, thr, 'x')} {x0} - {x1} {x0} - / {gcor} pow"
    else:
        lin = f'x {gcor} pow'

    if aka_expr_available:
        header = f'{header} {lin} LIN!'
        lin = 'LIN@'

    expr = f'{lin} {c.k0} {c.phi} / <= {lin} {c.phi} * {lin} 1 {c.gamma} / pow {c.alpha} 1 + * {c.alpha} - ?'

    return _clamp_converted(clip, header, expr, curve, sigmoid)
