from __future__ import annotations

from functools import partial
from math import log2
from typing import Callable, Iterable, Literal, Sequence, Type, overload

import vapoursynth as vs
from vsaa import Znedi3
from vsexprtools.util import normalise_seq
from vskernels import Catrom, Kernel, Spline144, get_kernel, get_prop
from vskernels.kernels.abstract import Scaler
from vsmask.edge import EdgeDetect
from vsutil import depth, get_depth, join, split

from .mask import descale_detail_mask
from .scale import scale_var_clip
from .types import CreditMaskT, DescaleAttempt

core = vs.core


__all__ = [
    'get_select_descale',
    'descale',
]


def get_select_descale(
    clip: vs.VideoNode, descale_attempts: list[DescaleAttempt], threshold: float = 0.0
) -> tuple[Callable[[list[vs.VideoFrame], int], vs.VideoNode], list[vs.VideoNode]]:
    clips_by_reskern = {
        attempt.da_hash: attempt
        for attempt in descale_attempts
    }

    attempts_by_idx = list(clips_by_reskern.values())

    diff_clips = [attempt.diff for attempt in attempts_by_idx]

    n_clips = len(diff_clips)

    def _get_descale_score(plane_averages: list[float], i: int) -> float:
        height_log = log2(clip.height - attempts_by_idx[i].resolution.height)
        pstats_avg = round(1 / max(plane_averages[i], 1e-12))

        return height_log * pstats_avg ** 0.2  # type: ignore

    def _parse_attemps(f: list[vs.VideoFrame]) -> tuple[vs.VideoNode, list[float], int]:
        plane_averages = [get_prop(frame, 'PlaneStatsAverage', float) for frame in f]

        best_res = max(range(n_clips), key=partial(_get_descale_score, plane_averages))

        best_attempt = attempts_by_idx[best_res]

        return best_attempt.descaled, plane_averages, best_res

    if threshold == 0:
        def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
            return _parse_attemps(f)[0]
    else:
        def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
            best_attempt, plane_averages, best_res = _parse_attemps(f)

            if plane_averages[best_res] > threshold:
                return clip

            return best_attempt

    return _select_descale, diff_clips


@overload
def descale(  # type: ignore
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None,
    height: int | Iterable[int] = 720,
    upscaler: Scaler | None = Znedi3(),
    kernels: Kernel | Type[Kernel] | str | Sequence[Kernel | Type[Kernel] | str] = Catrom(),
    thr: float = 0.0, shift: tuple[float, float] = (0, 0),
    mask: CreditMaskT | bool = descale_detail_mask,
    show_mask: Literal[False] = False
) -> vs.VideoNode:
    ...


@overload
def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None,
    height: int | Iterable[int] = 720,
    upscaler: Scaler | None = Znedi3(),
    kernels: Kernel | Type[Kernel] | str | Sequence[Kernel | Type[Kernel] | str] = Catrom(),
    thr: float = 0.0, shift: tuple[float, float] = (0, 0),
    mask: CreditMaskT | Literal[True] = descale_detail_mask,
    show_mask: Literal[True] = True
) -> tuple[vs.VideoNode, vs.VideoNode]:
    ...


def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None,
    height: int | Iterable[int] = 720,
    upscaler: Scaler | None = Znedi3(),
    kernels: Kernel | Type[Kernel] | str | Sequence[Kernel | Type[Kernel] | str] = Catrom(),
    thr: float = 0.0, shift: tuple[float, float] = (0, 0),
    mask: CreditMaskT | bool = descale_detail_mask,
    show_mask: bool = False
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
            clip_y, width, height, shift, kernel,
            descale_attempt_idx=i, descale_height=height, descale_kernel=kernel.__class__.__name__
        )
        for i, (kernel, (width, height)) in enumerate(kernel_combinations)
    ]

    if len(descale_attempts) > 1:
        var_res_clip = core.std.Splice([
            clip_y.std.BlankClip(length=len(clip_y) - 1, keep=True),
            clip_y.std.BlankClip(length=1, width=clip_y.width + 1, keep=True)
        ], mismatch=True)

        select_partial, prop_clips = get_select_descale(clip_y, descale_attempts, thr)

        descaled = var_res_clip.std.FrameEval(select_partial, prop_clips)
    else:
        descaled = descale_attempts[0].descaled

    if upscaler is None:
        upscaled = descaled
    else:
        upscaled = scale_var_clip(descaled, clip_y.width, clip_y.height, scaler=upscaler)

    if mask:
        if len(kernel_combinations) == 1:
            rescaled = descale_attempts[0].rescaled
        else:
            rescaled = clip_y.std.FrameEval(
                lambda f, n: descale_attempts[f.props.descale_attempt_idx].rescaled, descaled
            )

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
