from vstools import (CustomValueError, FuncExceptT, KwargsT,
                     check_variable_resolution, get_h, get_w, vs)


__all__ = [
    "get_match_centers_scaling"
]


def get_match_centers_scaling(
    clip: vs.VideoNode,
    target_width: int | None = None,
    target_height: int | None = 720,
    func_except: FuncExceptT | None = None
) -> KwargsT:
    """
    Convenience function to calculate the native resolution for sources that were upsampled
    using the "match centers" model as opposed to the more common "match edges" models.

    While match edges will align the edges of the outermost pixels in the target image,
    match centers will instead align the *centers* of the outermost pixels.

    Here's a visual example for a 3x1 image upsampled to 7x1:

        * Match edges:

    +-------------+-------------+-------------+
    |      ·      |      ·      |      ·      |
    +-------------+-------------+-------------+
    ↓                                         ↓
    +-----+-----+-----+-----+-----+-----+-----+
    |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |
    +-----+-----+-----+-----+-----+-----+-----+

        * Match centers:

    +-----------------+-----------------+-----------------+
    |        ·        |        ·        |        ·        |
    +-----------------+-----------------+-----------------+
             ↓                                   ↓
          +-----+-----+-----+-----+-----+-----+-----+
          |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |
          +-----+-----+-----+-----+-----+-----+-----+

    For a more detailed explanation, refer to this page: `<https://entropymine.com/imageworsener/matching/>`.

    The formula for calculating values we can use during desampling is simple:

    * width: clip.width * (target_width - 1) / (clip.width - 1)
    * height: clip.height * (target_height - 1) / (clip.height - 1)

    Example usage:

    .. code-block:: python

        >>> from vodesfunc import DescaleTarget
        >>> ...
        >>> native_res = get_match_centers_scaling(src, 1280, 720)
        >>> rescaled = DescaleTarget(kernel=Catrom, upscaler=Waifu2x, downscaler=Hermite(linear=True), **native_res)

    The output is meant to be passed to `vodesfunc.DescaleTarget` as keyword arguments,
    but it may also apply to other functions that require similar parameters.

    :param clip:            The clip to base the calculations on.
    :param target_width:    Target width for the descale. This should probably be equal to the base width.
                            If not provided, this value is calculated using the `target_height`.
                            Default: None.
    :param target_height:   Target height for the descale. This should probably be equal to the base height.
                            If not provided, this value is calculated using the `target_width`.
                            Default: 720.
    :param func_except:     Function returned for custom error handling.
                            This should only be set by VS package developers.

    :return:                A dictionary with the keys, {width, height, base_width, base_height},
                            which can be passed directly to `vodesfunc.DescaleTarget` or similar functions.
    """

    func = func_except or get_match_centers_scaling

    if target_width is None and target_height is None:
        raise CustomValueError("Either `target_width` or `target_height` must be a positive integer.", func)

    if target_width is not None and (not isinstance(target_width, int) or target_width <= 0):
        raise CustomValueError("`target_width` must be a positive integer or None.", func)

    if target_height is not None and (not isinstance(target_height, int) or target_height <= 0):
        raise CustomValueError("`target_height` must be a positive integer or None.", func)

    check_variable_resolution(clip, func)

    if target_height is None:
        target_height = get_h(target_width, clip, 1)  # type:ignore[arg-type]
    elif target_width is None:
        target_width = get_w(target_height, clip, 1)

    width = clip.width * (target_width - 1) / (clip.width - 1)  # type:ignore[operator]
    height = clip.height * (target_height - 1) / (clip.height - 1)

    return KwargsT(width=width, height=height, base_width=target_width, base_height=target_height)
