from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from vstools import CustomStrEnum

__all__ = [
    'ShaderFileBase',
    'ShaderFileCustom'
]

if TYPE_CHECKING:
    from .shaders import ShaderFile

    class ShaderFileCustomBase:
        CUSTOM: ShaderFileCustom

    class ShaderFileBase(ShaderFileCustomBase, CustomStrEnum):
        ...

    class ShaderFileCustom(ShaderFile):  # type: ignore
        def __call__(self, name: str | Path) -> Path:
            ...
else:
    ShaderFileBase = CustomStrEnum
    ShaderFileCustom = CustomStrEnum
