from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vskernels import Catrom, Kernel, Scaler, ScalerT, KernelT
from vstools import (
    CustomRuntimeError, FileWasNotFoundError, core, expect_bits, get_user_data_dir, get_video_format, get_y,
    inject_self, join, vs, CustomNotImplementedError
)

from .base import ShaderFileBase, ShaderFileCustom
from .helpers import GenericScaler

__all__ = [
    'PlaceboShader',

    'ShaderFile',

    'FSRCNNXShader', 'FSRCNNXShaderT'
]


class PlaceboShaderMeta(GenericScaler):
    shader_file: str | Path | ShaderFile


@dataclass
class PlaceboShaderBase(PlaceboShaderMeta):
    chroma_loc: int | None = field(default=None, kw_only=True)
    matrix: int | None = field(default=None, kw_only=True)
    trc: int | None = field(default=None, kw_only=True)
    linearize: int | None = field(default=None, kw_only=True)
    sigmoidize: int | None = field(default=None, kw_only=True)
    sigmoid_center: float | None = field(default=None, kw_only=True)
    sigmoid_slope: float | None = field(default=None, kw_only=True)
    lut_entries: int | None = field(default=None, kw_only=True)
    antiring: float | None = field(default=None, kw_only=True)
    filter_shader: str | None = field(default=None, kw_only=True)
    clamp: float | None = field(default=None, kw_only=True)
    blur: float | None = field(default=None, kw_only=True)
    taper: float | None = field(default=None, kw_only=True)
    radius: float | None = field(default=None, kw_only=True)
    param1: float | None = field(default=None, kw_only=True)
    param2: float | None = field(default=None, kw_only=True)

    scaler: ScalerT = field(default=Catrom, kw_only=True)
    shifter: KernelT | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()

        self._scaler = Scaler.ensure_obj(self.scaler, self.__class__)
        self._shifter = Kernel.ensure_obj(
            self.shifter or (self._scaler if isinstance(self._scaler, Kernel) else Catrom), self.__class__
        )

        if not hasattr(self, 'shader_file'):
            raise CustomRuntimeError('You must specify a "shader_file"!', self.__class__)

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        clip, _ = expect_bits(clip, 16)

        fmt = get_video_format(clip)

        if fmt.num_planes == 1:
            if width > clip.width or height > clip.height:
                clip = clip.resize.Point(format=vs.YUV444P16)
            else:
                for div in (4, 2):
                    if width % div == 0 and height % div == 0:
                        blank = core.std.BlankClip(clip, clip.width // div, clip.height // div, vs.GRAY16)
                        break
                else:
                    blank = clip.std.BlankClip(vs.GRAY16)

                clip = join(clip, blank, blank)

        kwargs |= {
            'shader': self.shader_file.path if isinstance(self.shader_file, ShaderFile) else str(self.shader_file),
            'chroma_loc': self.chroma_loc, 'matrix': self.matrix,
            'trc': self.trc, 'linearize': self.linearize,
            'sigmoidize': self.sigmoidize, 'sigmoid_center': self.sigmoid_center, 'sigmoid_slope': self.sigmoid_slope,
            'lut_entries': self.lut_entries,
            'antiring': self.antiring, 'filter': self.filter_shader, 'clamp': self.clamp,
            'blur': self.blur, 'taper': self.taper, 'radius': self.radius,
            'param1': self.param1, 'param2': self.param2,
        } | kwargs | {
            'width': clip.width * ceil(width / clip.width),
            'height': clip.height * ceil(height / clip.height)
        }

        if not kwargs['filter']:
            kwargs['filter'] = 'box' if fmt.num_planes == 1 else 'ewa_lanczos'

        if not Path(kwargs['shader']).exists():
            try:
                kwargs['shader'] = str(ShaderFile.CUSTOM(kwargs['shader']))
            except FileWasNotFoundError:
                ...

        clip = clip.placebo.Shader(**kwargs)

        if fmt.num_planes == 1:
            clip = get_y(clip)

        if (clip.width, clip.height) != (width, height):
            clip = self._scaler.scale(clip, width, height)

        if shift != (0, 0):
            if self.shifter:
                clip = self._shifter.shift(clip, shift)
            else:
                clip = self._kernel.shift(clip, shift)

        return self._kernel.resample(clip, fmt)


@dataclass
class PlaceboShader(PlaceboShaderBase):
    shader_file: str | Path


class ShaderFile(ShaderFileBase):
    if not TYPE_CHECKING:
        CUSTOM = 'custom'

    FSRCNNX_x8 = 'FSRCNNX_x2_8-0-4-1.glsl'
    FSRCNNX_x16 = 'FSRCNNX_x2_16-0-4-1.glsl'
    FSRCNNX_x56 = 'FSRCNNX_x2_56-16-4-1.glsl'

    SSIM_DOWNSCALER = 'SSimDownscaler.glsl'
    SSIM_SUPERSAMPLER = 'SSimSuperRes.glsl'

    def __call__(self: ShaderFileCustom, file_name: str | Path) -> Path:  # type: ignore
        if self is not ShaderFile.CUSTOM:
            raise CustomNotImplementedError

        in_cwd = Path.cwd() / file_name

        if in_cwd.is_file():
            return in_cwd

        asset_dir = Path.cwd() / '_assets' / file_name

        if asset_dir.is_file():
            return asset_dir

        mpv_dir = get_user_data_dir().parent / 'Roaming' / 'mpv' / 'shaders' / file_name

        if mpv_dir.is_file():
            return mpv_dir

        raise FileWasNotFoundError(f'"{file_name}" could not be found!', str(ShaderFile.CUSTOM))

    @cached_property
    def path(self) -> Path:
        if self is ShaderFile.CUSTOM:
            raise CustomNotImplementedError

        return Path(__file__).parent / 'shaders' / self.value


class FSRCNNXShader(PlaceboShaderBase):
    shader_file = ShaderFile.FSRCNNX_x56

    @dataclass
    class x8(PlaceboShaderBase):
        shader_file = ShaderFile.FSRCNNX_x8

    @dataclass
    class x16(PlaceboShaderBase):
        shader_file = ShaderFile.FSRCNNX_x16

    @dataclass
    class x56(PlaceboShaderBase):
        shader_file = ShaderFile.FSRCNNX_x56


FSRCNNXShaderT = type[PlaceboShaderBase] | PlaceboShaderBase  # type: ignore
