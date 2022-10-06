from __future__ import annotations

from functools import partial
from itertools import groupby
from math import log2
from typing import Callable, Iterable, Literal, Sequence, Type, cast, overload

from vsaa import Nnedi3
from vskernels import Catrom, Kernel, Spline144
from vskernels.kernels.abstract import Scaler
from vsmask.edge import EdgeDetect
from vstools import core, depth, get_depth, get_h, get_prop, get_w, join, normalize_seq, split, vs

from .mask import descale_detail_mask
from .scale import scale_var_clip
from .types import CreditMaskT, DescaleAttempt, DescaleMode, DescaleResult, PlaneStatsKind, _DescaleTypeGuards

__all__ = [
    'get_select_descale', 'descale'
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
