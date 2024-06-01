from dataclasses import dataclass
from typing import Any, ClassVar
from vskernels import KernelT, Kernel, Bilinear
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
    KwargsT,
    get_y,
)

from .helpers import GenericScaler

__all__ = ["GenericOnnxScaler", "autoselect_backend", "ArtCNN"]


@dataclass
class GenericOnnxScaler(GenericScaler):
    """Generic scaler class for an onnx model."""

    model: SPathLike | None = None
    """Path to the model."""
    backend: Any | None = None
    """
    vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.\n
    In order of trt > cuda > directml > nncn > cpu.
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

        wclip, _ = expect_bits(clip, 32)

        from vsmlrt import inference, calc_tilesize, init_backend

        if self.overlap is None:
            overlap_w = overlap_h = 8
        else:
            overlap_w, overlap_h = (self.overlap, self.overlap) if isinstance(self.overlap, int) else self.overlap

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
            raise CustomValueError(f"Tile size must be divisible by 1 ({tile_w}, {tile_h})", self)

        backend = init_backend(backend=self.backend, trt_opt_shapes=(tile_w, tile_h))

        scaled = inference(
            wclip,
            network_path=str(SPath(self.model).resolve()),
            backend=backend,
            overlap=(overlap_w, overlap_h),
            tilesize=(tile_w, tile_h),
        )
        return self._finish_scale(scaled, clip, width, height, shift)


def autoselect_backend(trt_args: KwargsT = {}, **kwargs: Any) -> Any:
    from vsmlrt import Backend
    import os

    fp16 = kwargs.pop("fp16", True)

    cuda = get_nvidia_version() is not None
    if cuda:
        if hasattr(core, "trt"):
            kwargs.update(trt_args)
            return Backend.TRT(fp16=fp16, **trt_args)
        elif hasattr(core, "ort"):
            return Backend.ORT_CUDA(fp16=fp16, **kwargs)
        else:
            return Backend.OV_GPU(fp16=fp16, **kwargs)
    else:
        if hasattr(core, "ort") and os.name == "nt":
            return Backend.ORT_DML(fp16=fp16, **kwargs)
        elif hasattr(core, "ncnn"):
            return Backend.NCNN_VK(fp16=fp16, **kwargs)

        return Backend.ORT_CPU(fp16=fp16, **kwargs) if hasattr(core, "ort") else Backend.OV_CPU(fp16=fp16, **kwargs)


class _BaseArtCNN:
    _model: ClassVar[int]
    _func = "ArtCNN"


@dataclass
class BaseArtCNN(_BaseArtCNN, GenericScaler):
    backend: Any | None = None
    """
    vs-mlrt backend. Will attempt to autoselect the most suitable one with fp16=True if None.\n
    In order of trt > cuda > directml > nncn > cpu.
    """
    chroma_scaler: KernelT | None = None
    """
    Scaler to upscale the chroma with.\n
    Necessary if you're trying to use one of the chroma models but aren't passing a 444 clip.\n
    Bilinear is probably the safe option to use.
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
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        clip_format = get_video_format(clip)
        chroma_model = self._model in range(4, 6)

        # The chroma models aren't supposed to change the video dimensions and API wise this is more comfortable.
        if width is None or height is None:
            if chroma_model:
                width = clip.width
                height = clip.height
            else:
                raise CustomValueError("You have to pass height and width if not using a chroma model.", self._func)

        if chroma_model and clip_format.color_family != vs.YUV:
            raise CustomValueError("ArtCNN Chroma models need YUV input.", self._func)

        if not chroma_model and clip_format.color_family not in (vs.YUV, vs.GRAY):
            raise CustomValueError("Regular ArtCNN models need YUV or GRAY input.", self._func)

        if chroma_model and (clip_format.subsampling_h != 0 or clip_format.subsampling_w != 0):
            if self.chroma_scaler is None:
                raise CustomValueError(
                    "ArtCNN needs a non subsampled clip. Either pass one or set `chroma_scaler`.", self._func
                )

            clip = Kernel.ensure_obj(self.chroma_scaler).resample(
                clip, clip_format.replace(subsampling_h=0, subsampling_w=0)
            )

        wclip = get_y(clip) if not chroma_model else clip

        if self.backend is None:
            self.backend = autoselect_backend()

        from vsmlrt import ArtCNN as mlrt_ArtCNN, ArtCNNModel

        scaled = mlrt_ArtCNN(
            depth(wclip, 32), self.tiles, self.tilesize, self.overlap, ArtCNNModel(self._model), backend=self.backend
        )

        return self._finish_scale(scaled, wclip, width, height, shift)


class ArtCNN(BaseArtCNN):
    _model = 2

    class C4F32(BaseArtCNN):
        _model = 0

    class C4F32_DS(BaseArtCNN):
        _model = 1

    class C16F64(BaseArtCNN):
        _model = 2

    class C16F64_DS(BaseArtCNN):
        _model = 3

    class C4F32_Chroma(BaseArtCNN):
        _model = 4

    class C16F64_Chroma(BaseArtCNN):
        _model = 5
