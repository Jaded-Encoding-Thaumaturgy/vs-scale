from dataclasses import dataclass
from typing import Any
from vstools import (
    SPath,
    SPathLike,
    inject_self,
    vs,
    core,
    get_video_format,
    expect_bits,
    depth,
    CustomValueError,
    get_nvidia_version,
)

from .helpers import GenericScaler

__all__ = ["GenericOnnxScaler"]


@dataclass
class GenericOnnxScaler(GenericScaler):
    """Generic scaler class for an onnx model."""

    model: SPathLike | None = None
    """Path to the model."""
    backend: Any | None = None
    """
    vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.\n
    In order of trt > cuda > nncn > cpu.
    """
    tiles: int | tuple[int, int] | None = None
    """Splits up the frame into multiple tiles. Helps if you're lacking in vram but models may behave differently."""

    tilesize: int | tuple[int, int] | None = None
    overlap: int | tuple[int, int] | None = None

    _static_kernel_radius = 2

    @inject_self
    def scale(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        if self.backend is None:
            self.backend = autoselect_backend()

        clip_format = get_video_format(clip)
        if clip_format.subsampling_h != 0 or clip_format.subsampling_w != 0:
            raise CustomValueError("This scaler requires non subsampled input!", self)

        wclip, og_depth = expect_bits(clip, 32)

        from vsmlrt import inference, calc_tilesize, init_backend

        if self.overlap is None:
            overlap_w = overlap_h = 8
        else:
            overlap_w, overlap_h = (
                (self.overlap, self.overlap)
                if isinstance(self.overlap, int)
                else self.overlap
            )

        (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
            tiles=self.tiles,
            tilesize=self.tilesize,
            width=wclip.width,
            height=wclip.height,
            multiple=1,
            overlap_w=overlap_w,
            overlap_h=overlap_h,
        )

        if tile_w % 1 != 0 or tile_h % 1 != 0:
            raise CustomValueError(
                f"Tile size must be divisible by 1 ({tile_w}, {tile_h})", self
            )

        backend = init_backend(backend=self.backend, trt_opt_shapes=(tile_w, tile_h))

        scaled = inference(
            wclip,
            network_path=str(SPath(self.model).resolve()),
            backend=backend,
            overlap=(overlap_w, overlap_h),
            tilesize=(tile_w, tile_h),
        )
        scaled = self._finish_scale(scaled, wclip, width, height, shift)
        return depth(scaled, og_depth)


def autoselect_backend() -> Any:
    from vsmlrt import Backend

    cuda = get_nvidia_version() is not None
    if cuda:
        if hasattr(core, "trt"):
            return Backend.TRT(fp16=True)
        elif hasattr(core, "ort"):
            return Backend.ORT_CUDA(fp16=True)
        else:
            return Backend.OV_GPU(fp16=True)
    else:
        if hasattr(core, "ncnn"):
            return Backend.NCNN_VK(fp16=True)
        else:
            return (
                Backend.ORT_CPU(fp16=True)
                if hasattr(core, "ort")
                else Backend.OV_CPU(fp16=True)
            )
