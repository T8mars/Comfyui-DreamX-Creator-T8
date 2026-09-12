import torch

from comfy_nodes.nodes import DreamXFirstFrameAVLatent, DreamXSplitAVLatent


class FakeWanVAE:
    def encode(self, images):
        return torch.zeros(
            (1, 48, 1, images.shape[1] // 16, images.shape[2] // 16),
            dtype=torch.float32,
        )


def test_first_frame_node_builds_native_nested_latent_and_metadata():
    image = torch.zeros((1, 720, 1280, 3), dtype=torch.float32)
    conditioning = [[torch.zeros(1, 512, 4096), {}]]
    output = DreamXFirstFrameAVLatent.execute(
        conditioning, conditioning, FakeWanVAE(), image,
        duration=5.0, fps=24.0, target_spatial_tokens=880, batch_size=1,
    )
    positive, negative, latent, width, height, frames, fps = output.result
    assert (width, height, frames) == (1248, 704, 117)
    assert fps == 24.0
    assert positive[0][1]["dreamx_fps"] == negative[0][1]["dreamx_fps"] == 24.0
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 48, 30, 44, 78)
    assert audio.shape == (1, 128, 244)
    assert latent["dreamx_audio_samples"] == 234000
    video_out, audio_out = DreamXSplitAVLatent.execute(latent).result
    assert video_out["samples"].shape == video.shape
    assert audio_out["samples"].shape == audio.shape
