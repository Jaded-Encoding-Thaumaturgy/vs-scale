from __future__ import annotations

from functools import partial
from itertools import groupby
from math import log2
from typing import Callable, Iterable, Literal, Sequence, Type, cast, overload

from vsaa import Eedi3, Nnedi3, SuperSampler
from vskernels import Catrom, Kernel, KernelT, Scaler, Spline144
from vsmask.edge import EdgeDetect
from vstools import (
    check_variable, core, depth, get_depth, get_h, get_prop, get_w, get_y, iterate, join, normalize_seq, scale_thresh,
    split, vs
)

from .mask import descale_detail_mask
from .scale import SSIM, scale_var_clip
from .types import CreditMaskT, DescaleAttempt, DescaleMode, DescaleResult, PlaneStatsKind, _DescaleTypeGuards

__all__ = [
    'get_select_descale', 'descale',

    'mixed_rescale'
]


def get_select_descale(
    clip: vs.VideoNode, descale_attempts: list[DescaleAttempt], mode: DescaleMode
) -> tuple[Callable[[list[vs.VideoFrame], int], vs.VideoNode], list[vs.VideoNode]]:
    clips_by_reskern = {
        attempt.da_hash: attempt
        for attempt in descale_attempts
    }

    curr_clip = clip
    main_kernel = descale_attempts[0].kernel.__class__.__name__
    attempts_by_idx = list(clips_by_reskern.values())

    threshold = mode.thr
    res_operator = mode.res_op
    diff_operator = mode.diff_op

    res_props_key = DescaleMode.PlaneDiff.prop_value(PlaneStatsKind.AVG)
    diff_prop_key = DescaleMode.KernelDiff.prop_value(PlaneStatsKind.DIFF)

    diff_clips = [attempt.diff for attempt in attempts_by_idx]

    def _get_descale_score(diff_vals: list[float], i: int) -> float:
        height_log = log2(curr_clip.height - attempts_by_idx[i].resolution.height)
        pstats_avg = round(1 / max(diff_vals[i], 1e-12))

        return height_log * pstats_avg ** 0.2  # type: ignore

    def _parse_attemps(f: list[vs.VideoFrame], indices: list[int]) -> tuple[vs.VideoNode, list[float], int]:
        diff_vals = [get_prop(frame, res_props_key, float) for frame in f]

        best_res = res_operator(indices, key=partial(_get_descale_score, diff_vals))

        best_attempt = attempts_by_idx[best_res]

        return best_attempt.descaled, diff_vals, best_res

    if mode.is_average:
        clips_indices = list(range(len(diff_clips)))

        if threshold <= 0.0:
            def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
                return _parse_attemps(f, clips_indices)[0]
        else:
            def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
                best_attempt, plane_averages, best_res = _parse_attemps(f, clips_indices)

                if plane_averages[best_res] > threshold:
                    return curr_clip

                return best_attempt
    elif mode.is_kernel_diff:
        group_by_kernel = {
            key: list(grouped) for key, grouped in groupby(
                enumerate(attempts_by_idx), lambda x: x[1].kernel.__class__.__name__
            )
        }

        if len(group_by_kernel) < 2:
            raise ValueError(
                'get_select_descale: With KernelDiff mode you need to specify at least two kernels!\n'
                '(First will be the main kernel, others will be compared to it)'
            )

        kernel_indices = {
            name: [x[0] for x in attempts] for name, attempts in group_by_kernel.items()
        }
        other_kernels = {
            key: val for key, val in kernel_indices.items() if key != main_kernel
        }

        main_kernel_indices = list(kernel_indices[main_kernel])
        other_kernel_indices = list(other_kernels.values())
        other_kernel_enum_indices = list(enumerate(other_kernels.values()))

        main_clips_indices = list(range(len(main_kernel_indices)))
        other_clips_indices = [
            list(range(len(kernel_indices)))
            for kernel_indices in other_kernel_indices
        ]
        comp_other_indices = list(range(len(other_clips_indices)))

        def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
            main_clip, _, main_best_val_idx = _parse_attemps(
                [f[i] for i in main_kernel_indices], main_clips_indices
            )

            other_diffs_parsed = [
                _parse_attemps([f[i] for i in indices], other_clips_indices[j])
                for j, indices in other_kernel_enum_indices
            ]

            other_best_idx = diff_operator(
                comp_other_indices, key=lambda i: other_diffs_parsed[i][1][other_diffs_parsed[i][2]]
            )
            other_clip, _, other_best_val_idx = other_diffs_parsed[other_best_idx]

            main_value = get_prop(f[main_kernel_indices[main_best_val_idx]], diff_prop_key, float)
            other_value = get_prop(f[other_kernel_indices[other_best_idx][other_best_val_idx]], diff_prop_key, float)

            if other_value - threshold > main_value:
                return main_clip

            return other_clip

    else:
        raise ValueError('get_select_descale: incorrect descale mode specified!')

    return _select_descale, diff_clips


DescKernelsT = Kernel | Type[Kernel] | str | Sequence[Kernel | Type[Kernel] | str]


@overload
def descale(  # type: ignore
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: DescKernelsT = Catrom(), upscaler: Scaler | None = Nnedi3(),
    mask: CreditMaskT | Literal[False] = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: Kernel = Spline144(),
    result: Literal[False] = False
) -> vs.VideoNode:
    ...


@overload
def descale(  # type: ignore
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: DescKernelsT = Catrom(), upscaler: Scaler = Nnedi3(),
    mask: CreditMaskT = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: Kernel = Spline144(),
    result: Literal[True] = True
) -> _DescaleTypeGuards.UpscalerNotNoneMaskNotNone:
    ...


@overload
def descale(  # type: ignore
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: DescKernelsT = Catrom(), upscaler: Scaler = Nnedi3(),
    mask: Literal[False] = False,
    mode: DescaleMode = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: Kernel = Spline144(),
    result: Literal[True] = True
) -> _DescaleTypeGuards.UpscalerNotNoneMaskIsNone:
    ...


@overload
def descale(  # type: ignore
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: DescKernelsT = Catrom(), upscaler: None = None,
    mask: CreditMaskT = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: Kernel = Spline144(),
    result: Literal[True] = True
) -> _DescaleTypeGuards.UpscalerIsNoneMaskNotNone:
    ...


@overload
def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: DescKernelsT = Catrom(), upscaler: None = None,
    mask: Literal[False] = False,
    mode: DescaleMode = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: Kernel = Spline144(),
    result: Literal[True] = True
) -> _DescaleTypeGuards.UpscalerIsNoneMaskIsNone:
    ...


@overload
def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: DescKernelsT = Catrom(), upscaler: Scaler | None = Nnedi3(),
    mask: CreditMaskT | Literal[False] = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: Kernel = Spline144(),
    result: Literal[True] = True
) -> DescaleResult:
    ...


def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: DescKernelsT = Catrom(), upscaler: Scaler | None = Nnedi3(),
    mask: CreditMaskT | Literal[False] = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: Kernel = Spline144(),
    result: bool = False
) -> vs.VideoNode | DescaleResult:
    assert clip.format

    if isinstance(height, int):
        heights = [height]
    else:
        heights = list(height)

    if width is None:
        widths = [get_w(h, clip) for h in heights]
    elif isinstance(width, int):
        widths = [width]
    else:
        widths = list(width)

    if dst_width is None and dst_height:
        dest_width, dest_height = get_w(dst_height, clip), dst_height
    elif dst_height is None and dst_width:
        dest_width, dest_height = dst_width, get_h(dst_width, clip)
    else:
        dest_width, dest_height = clip.width, clip.height

    if not isinstance(kernels, Sequence):
        kernels = [kernels]

    norm_resolutions = list(zip(widths, heights))
    norm_kernels = [
        kernel if isinstance(kernel, Kernel) else Kernel.from_param(kernel)()
        for kernel in kernels
    ]

    if len(widths) != len(heights):
        raise ValueError("descale: Number of heights and widths specified mismatch!")

    if not norm_kernels:
        raise ValueError("descale: You must specify at least one kernel!")

    work_clip, *chroma = split(clip)

    bit_depth = get_depth(clip)
    clip_y = work_clip.resize.Point(format=vs.GRAYS)

    max_kres_len = max(len(norm_kernels), len(norm_resolutions))

    kernel_combinations = list[tuple[Kernel, tuple[int, int]]](zip(*(
        normalize_seq(x, max_kres_len) for x in (norm_kernels, norm_resolutions)  # type: ignore
    )))

    descale_attempts = [
        DescaleAttempt.from_args(
            clip_y, width, height, shift, kernel, mode,
            descale_attempt_idx=i, descale_height=height, descale_kernel=kernel.__class__.__name__
        )
        for i, (kernel, (width, height)) in enumerate(kernel_combinations)
    ]

    if len(descale_attempts) > 1:
        var_res_clip = core.std.Splice([
            clip_y.std.BlankClip(length=len(clip_y) - 1, keep=True),
            clip_y.std.BlankClip(length=1, width=clip_y.width + 1, keep=True)
        ], mismatch=True)

        select_partial, prop_clips = get_select_descale(clip_y, descale_attempts, mode)

        descaled = var_res_clip.std.FrameEval(select_partial, prop_clips)
    else:
        descaled = descale_attempts[0].descaled

    if len(kernel_combinations) == 1:
        rescaled = descale_attempts[0].rescaled
    else:
        rescaled = clip_y.std.FrameEval(
            lambda f, n: descale_attempts[cast(int, f.props.descale_attempt_idx)].rescaled, descaled
        )

    clip_y = depth(clip_y, bit_depth)
    descaled = depth(descaled, bit_depth)
    rescaled = depth(rescaled, bit_depth)

    upscaled = None
    if upscaler:
        upscaled = scale_var_clip(descaled, dest_width, dest_height, scaler=upscaler)
        upscaled = depth(upscaled, bit_depth)

    if mask:
        if isinstance(mask, EdgeDetect):
            mask = mask.edgemask(clip_y)
        elif callable(mask):
            mask = mask(clip, rescaled)

        assert isinstance(mask, vs.VideoNode)

        mask = depth(mask, bit_depth)

        if upscaled:
            mask = scaler.scale(mask, dest_width, dest_height)
            clip_y = scaler.scale(clip_y, dest_width, dest_height)
            upscaled = upscaled.std.MaskedMerge(clip_y, mask)

    if upscaled:
        out = upscaled
    elif upscaler:
        out = descaled
    else:
        out = clip

    if chroma and upscaled and (clip.width, clip.height) == (dest_width, dest_height):
        out = join([upscaled, *chroma], clip.format.color_family)

    if result:
        return DescaleResult(
            descaled, rescaled, upscaled, mask if mask else None, descale_attempts, out  # type: ignore
        )

    return out


def mixed_rescale(
    clip: vs.VideoNode, width: None | int = None, height: int = 720,
    kernel: KernelT = Catrom,
    downscaler: Scaler | KernelT = SSIM,
    credit_mask: CreditMaskT | vs.VideoNode | None = descale_detail_mask, mask_thr: float = 0.05,
    mix_strength: float = 0.25, show_mask: bool | int = False,
    # Default settings set to match insaneAA as closely as reasonably possible
    eedi3: SuperSampler = Eedi3(
        alpha=0.2, beta=0.25, gamma=1000, nrad=2, mdis=20, sclip_aa=Nnedi3(nsize=0, nns=4, qual=2, pscrn=1)
    )
) -> vs.VideoNode:
    """
    Rewrite of InsaneAA to make it easier to use and maintain.
    Written by LightArrowsEXE, taken from lvsfunc.

    Descales and downscales the given clip and merges them together with a set strength.

    This can be useful for dealing with a source that you can't accurately descale,
    but you still want to force it. Not recommended to use it on everything, however.

    A string can be passed instead of a Kernel object if you want to use that.
    This gives you access to every kernel object in :py:mod:`vskernels`.
    For more information on what every kernel does, please refer to their documentation.

    :param clip:            Clip to process.
    :param width:           Upscale width. If None, determine from `height` (Default: None).
    :param height:          Upscale height (Default: 720).
    :param kernel:          py:class:`vskernels.Kernel` object used for the descaling.
                            This can also be the string name of the kernel
                            (Default: py:class:`vskernels.Catrom`).
    :param downscaler:      Kernel or custom scaler used to downscale the clip.
                            This can also be the string name of the kernel
                            (Default: py:func:`lvsfunc.scale.ssim_downsample`).
    :param credit_mask:     Function or mask clip used to mask detail. If ``None``, no masking.
                            Function must accept a clip and a reupscaled clip and return a mask.
                            (Default: :py:func:`lvsfunc.scale.descale_detail_mask`).
    :param mask_thr:        Binarization threshold for :py:func:`lvsfunc.scale.descale_detail_mask` (Default: 0.05).
    :param mix_strength:    Merging strength between the descaled and downscaled clip.
                            Stronger values will make the line-art look closer to the downscaled clip.
                            This can get pretty dangerous very quickly if you use a sharp ``downscaler``!
    :param show_mask:       Return the ``credit_mask``. If set to `2`, it will return the line-art mask instead.
    :param eedi3:           Eedi3 instance that will be used for supersampling.

    :return:                Rescaled clip with a downscaled clip merged with it and credits masked.

    :raises ValueError:     ``mask_thr`` is not between 0.0–1.0.
    """
    assert check_variable(clip, "mixed_rescale")

    if not 0 <= mask_thr <= 1:
        raise ValueError(f"mixed_rescale: '`mask_thr` must be between 0.0 and 1.0! Not {mask_thr}!'")

    width = width or get_w(height, clip.width / clip.height, 1)

    kernel = Kernel.ensure_obj(kernel)
    downscaler = Kernel.ensure_obj(downscaler)

    bits = get_depth(clip)
    clip_y = get_y(clip)

    line_mask = clip_y.std.Prewitt(scale=2).std.Maximum().std.Limiter()

    descaled = kernel.descale(clip_y, width, height)
    upscaled = kernel.scale(descaled, clip.width, clip.height)

    downscaled = downscaler.scale(clip_y, width, height)

    merged = core.akarin.Expr([descaled, downscaled], f'x {mix_strength} * y 1 {mix_strength} - * +')

    if isinstance(credit_mask, vs.VideoNode):
        detail_mask = depth(credit_mask, get_depth(clip))
    elif not credit_mask:
        detail_mask = clip_y.std.BlankClip(length=1) * clip.num_frames
    else:
        detail_mask = descale_detail_mask(clip_y, upscaled, threshold=scale_thresh(mask_thr, clip))
        detail_mask = iterate(detail_mask, core.std.Inflate, 2)
        detail_mask = iterate(detail_mask, core.std.Maximum, 2).std.Limiter()

    if show_mask == 2:
        return line_mask
    elif show_mask:
        return detail_mask

    double = eedi3.scale(merged, clip.width * 2, clip.height * 2)
    rescaled = SSIM.scale(double, clip.width, clip.height)

    rescaled = depth(rescaled, bits)

    masked = rescaled.std.MaskedMerge(clip_y, detail_mask)
    masked = clip_y.std.MaskedMerge(masked, line_mask)

    if clip.format.num_planes == 1:
        return masked

    return core.std.ShufflePlanes([masked, clip], planes=[0, 1, 2], colorfamily=vs.YUV)


# TODO: Write a function that checks every possible combination of B and C in bicubic
#       and returns a list of the results.
#       Possibly return all the frames in order of smallest difference to biggest.
#       Not reliable, but maybe useful as starting point.


# TODO: Write "multi_descale", a function that allows you to descale a frame twice,
#       like for example when the CGI in a show is handled in a different resolution
#       than the drawn animation.
