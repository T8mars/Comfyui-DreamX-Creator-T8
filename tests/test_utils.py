import pytest

from comfy_nodes.utils import compute_dynamic_resolution, snap_video_frames


def test_dynamic_resolution_stays_in_budget_for_16_by_9():
    height, width, tokens = compute_dynamic_resolution(720, 1280, 880)
    assert (height, width, tokens) == (704, 1248, 858)
    assert 0.95 * 880 <= tokens <= 880
    assert height % 32 == width % 32 == 0


@pytest.mark.parametrize(
    ("duration", "fps", "expected"),
    [(5.0, 24.0, 121), (1.0, 24.0, 25), (0.25, 24.0, 5)],
)
def test_frame_count_is_wan_temporal_stride_compatible(duration, fps, expected):
    frames = snap_video_frames(duration, fps)
    assert frames == expected
    assert (frames - 1) % 4 == 0
