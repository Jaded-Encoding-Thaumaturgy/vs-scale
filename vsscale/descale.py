from __future__ import annotations

from functools import partial
from itertools import groupby
from math import log2
from typing import Callable, Iterable, Literal, Sequence, Type, overload

import vapoursynth as vs
from vsaa import Nnedi3
from vsexprtools import normalise_seq
from vskernels import Catrom, Kernel, Spline144, get_kernel, get_prop
from vskernels.kernels.abstract import Scaler
from vsmask.edge import EdgeDetect
from vsutil import depth, get_depth, join, split

from .mask import descale_detail_mask
from .scale import scale_var_clip
from .types import CreditMaskT, DescaleAttempt, DescaleMode, PlaneStatsKind

core = vs.core


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

    res_props_key = DescaleMode.PlaneAverage.prop_value(PlaneStatsKind.AVG)
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


@overload
def descale(  # type: ignore
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None,
    height: int | Iterable[int] = 720,
    upscaler: Scaler | bool | None = Nnedi3(),
    kernels: Kernel | Type[Kernel] | str | Sequence[Kernel | Type[Kernel] | str] = Catrom(),
    shift: tuple[float, float] = (0, 0), mask: CreditMaskT | bool = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneAverage(0.0), show_mask: Literal[False] = False
) -> vs.VideoNode:
    ...


@overload
def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None,
    height: int | Iterable[int] = 720,
    upscaler: Scaler | Literal[True] | None = Nnedi3(),
    kernels: Kernel | Type[Kernel] | str | Sequence[Kernel | Type[Kernel] | str] = Catrom(),
    shift: tuple[float, float] = (0, 0), mask: CreditMaskT | Literal[True] = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneAverage(0.0), show_mask: Literal[True] = True
) -> tuple[vs.VideoNode, vs.VideoNode]:
    ...


def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None,
    height: int | Iterable[int] = 720,
    upscaler: Scaler | bool | None = Nnedi3(),
    kernels: Kernel | Type[Kernel] | str | Sequence[Kernel | Type[Kernel] | str] = Catrom(),
    shift: tuple[float, float] = (0, 0), mask: CreditMaskT | bool = descale_detail_mask,
    mode: DescaleMode = DescaleMode.PlaneAverage(0.0), show_mask: bool = False
) -> vs.VideoNode | tuple[vs.VideoNode, vs.VideoNode]:
    assert clip.format

    if isinstance(height, int):
        heights = [height]
    else:
        heights = list(height)

    if width is None:
        widths = [round(h * clip.width / clip.height) for h in heights]
    elif isinstance(width, int):
        widths = [width]
    else:
        widths = list(width)

    if not isinstance(kernels, Sequence):
        kernels = [kernels]

    norm_resolutions = list(zip(widths, heights))
    norm_kernels = [
        get_kernel(kernel)() if isinstance(kernel, str) else (
            kernel if isinstance(kernel, Kernel) else kernel()
        ) for kernel in kernels
    ]

    if len(widths) != len(heights):
        raise ValueError("descale: Number of heights and widths specified mismatch!")

    if not norm_kernels:
        raise ValueError("descale: You must specify at least one kernel!")

    work_clip, *chroma = split(clip)

    clip_y = work_clip.resize.Point(format=vs.GRAYS)

    max_kres_len = max(len(norm_kernels), len(norm_resolutions))

    kernel_combinations = list[tuple[Kernel, tuple[int, int]]](zip(*(
        normalise_seq(x, max_kres_len) for x in (norm_kernels, norm_resolutions)  # type: ignore
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

    if upscaler is False or mask:
        if len(kernel_combinations) == 1:
            rescaled = descale_attempts[0].rescaled
        else:
            rescaled = clip_y.std.FrameEval(
                lambda f, n: descale_attempts[f.props.descale_attempt_idx].rescaled, descaled
            )

    if upscaler is True:
        upscaler = Nnedi3()

    if upscaler is False:
        return rescaled

    if upscaler is None:
        upscaled = descaled
    else:
        upscaled = scale_var_clip(descaled, clip_y.width, clip_y.height, scaler=upscaler)

    if mask:
        if mask is True:
            mask = descale_detail_mask

        if isinstance(mask, EdgeDetect):
            mask = mask.edgemask(clip_y)
        elif callable(mask):
            mask = mask(clip_y, rescaled)

        if upscaler is None:
            mask = Spline144().scale(mask, upscaled.width, upscaled.height)
            clip_y = Spline144().scale(clip_y, upscaled.width, upscaled.height)

        upscaled = upscaled.std.MaskedMerge(clip_y, mask)

    upscaled = depth(upscaled, get_depth(clip))

    if not chroma:
        out = upscaled
    else:
        out = join([upscaled, *chroma], clip.format.color_family)

    if mask and show_mask:
        return out, mask  # type: ignore

    return out
