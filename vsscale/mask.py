import vapoursynth as vs
from vsexprtools import ExprOp, combine, expect_bits
from vsmask.edge import PrewittTCanny
from vsrgtools import removegrain
from vsutil import depth, get_y, iterate

core = vs.core


def descale_detail_mask(clip: vs.VideoNode, rescaled: vs.VideoNode, threshold: float = 0.05) -> vs.VideoNode:
    mask = core.akarin.Expr([get_y(clip), get_y(rescaled)], 'x y - abs')

    mask = mask.std.Binarize(threshold)

    mask = iterate(mask, core.std.Maximum, 4)
    mask = iterate(mask, core.std.Inflate, 2)

    return mask.std.Limiter()


def credit_mask(
    clip: vs.VideoNode, ref: vs.VideoNode, thr: int,
    blur: float | None = 1.65, prefilter: bool = True,
    expand: int = 8
) -> vs.VideoNode:
    from vardefunc.mask import Difference

    if blur is None or blur <= 0:
        blur_src, blur_ref = clip, ref
    else:
        blur_src = clip.bilateral.Gaussian(blur)
        blur_ref = ref.bilateral.Gaussian(blur)

    ed_mask = Difference().creditless(
        blur_src[0] + blur_src, blur_src, blur_ref,
        start_frame=0, thr=thr, prefilter=prefilter
    )

    bits, credit_mask = expect_bits(ed_mask)
    credit_mask = iterate(credit_mask, core.std.Minimum, 6)
    credit_mask = iterate(credit_mask, lambda x: core.std.Minimum(x).std.Maximum(), 8)
    if expand:
        credit_mask = iterate(credit_mask, core.std.Maximum, expand)
    credit_mask = credit_mask.std.Inflate().std.Inflate().std.Inflate()

    return credit_mask if bits == 16 else depth(credit_mask, bits)


def simple_detail_mask(
    clip: vs.VideoNode, sigma: float | None = None, rad: int = 3, brz_a: float = 0.025, brz_b: float = 0.045
) -> vs.VideoNode:
    from lvsfunc import range_mask, scale_thresh

    brz_a = scale_thresh(brz_a, clip)
    brz_b = scale_thresh(brz_b, clip)

    y = get_y(clip)

    blur = y.bilateral.Gaussian(sigma) if sigma else y

    mask_a = range_mask(blur, rad=rad).std.Binarize(brz_a)

    mask_b = PrewittTCanny().edgemask(blur).std.Binarize(brz_b)

    mask = combine([mask_a, mask_b])

    return removegrain(removegrain(mask, 22), 11).std.Limiter()


def multi_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> vs.VideoNode:
    general_mask = simple_detail_mask(clip, rad=1, brz_a=1, brz_b=24.3 * thr)

    return combine([
        combine([
            simple_detail_mask(clip, brz_a=1, brz_b=2 * thr),
            iterate(general_mask, core.std.Maximum, 3).std.Maximum().std.Inflate()
        ], ExprOp.MIN), general_mask.std.Maximum()
    ], ExprOp.MIN)
