import vapoursynth as vs

__all__ = [
    'merge_clip_props'
]


def merge_clip_props(*clips: vs.VideoNode, main_idx: int = 0) -> vs.VideoNode:
    if len(clips) == 1:
        return clips[0]

    def _merge_props(f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
        fdst = f[main_idx].copy()

        for i, frame in enumerate(f):
            if i == main_idx:
                continue

            fdst.props.update(frame.props)

        return fdst

    return clips[0].std.ModifyFrame(clips, _merge_props)
