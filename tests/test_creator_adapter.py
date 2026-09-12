import torch

import comfy.utils
from comfy.nested_tensor import NestedTensor

from comfy_nodes.creator_model import (
    DreamXCreatorBaseModel,
    DreamXCreatorDiffusion,
    time_shift_sigma,
)


class ToyJoint(torch.nn.Module):
    video_patch_size = (1, 2, 2)
    audio_patch_size = (1,)

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.last = None

    def forward(self, video, audio, dtype, enable_a2v, enable_v2a):
        self.last = (video, audio, enable_a2v.clone(), enable_v2a.clone())
        return {
            "video": torch.stack(video["x"]) * 2,
            "audio": torch.stack(audio["x"]) * 3,
        }


def test_adapter_builds_first_frame_zero_timestep_and_bridge_masks():
    joint = ToyJoint()
    adapter = DreamXCreatorDiffusion(joint, torch.float32)
    video = torch.ones(2, 48, 3, 4, 6)
    audio = torch.ones(2, 128, 9)
    context = torch.zeros(2, 512, 4096)
    out_video, out_audio = adapter(
        [video, audio], torch.tensor([750.0, 500.0]), context=context,
        dreamx_bridge=torch.tensor([[0.0], [1.0]]), dreamx_fps=24.0,
        transformer_options={"dreamx_enable_a2v": True, "dreamx_enable_v2a": False},
    )
    assert out_video.shape == video.shape
    assert out_audio.shape == audio.shape
    video_call, _, a2v, v2a = joint.last
    first_frame_tokens = 4 * 6 // 4
    assert torch.count_nonzero(video_call["t"][:, :first_frame_tokens]) == 0
    assert a2v.tolist() == [False, True]
    assert v2a.tolist() == [False, False]


def test_base_model_normalizes_only_video_and_roundtrips_packed_latent():
    model = DreamXCreatorBaseModel(ToyJoint(), torch.float32)
    video = torch.randn(1, 48, 2, 4, 4)
    audio = torch.randn(1, 128, 7)
    packed, shapes = comfy.utils.pack_latents([video, audio])
    model.latent_shapes = shapes
    normalized = model.process_latent_in(packed)
    restored = model.process_latent_out(normalized)
    torch.testing.assert_close(restored, packed)
    streams = comfy.utils.unpack_latents(normalized, shapes)
    torch.testing.assert_close(streams[1], audio)


def test_audio_stream_scaling_roundtrips_when_flow_shifts_differ():
    model = DreamXCreatorBaseModel(ToyJoint(), torch.float32)
    video = torch.randn(1, 48, 1, 2, 2)
    audio = torch.randn(1, 128, 3)
    packed, shapes = comfy.utils.pack_latents([video, audio])
    model.latent_shapes = shapes
    model.model_sampling.set_parameters(shift=5.0, audio_shift=2.5)
    normalized = model.process_latent_in(packed)
    normalized_audio = comfy.utils.unpack_latents(normalized, shapes)[1]
    torch.testing.assert_close(normalized_audio, audio * 2.0)
    torch.testing.assert_close(model.process_latent_out(normalized), packed)


def test_adapter_maps_independent_audio_schedule_and_velocity():
    joint = ToyJoint()
    adapter = DreamXCreatorDiffusion(joint, torch.float32)
    video = torch.ones(1, 48, 1, 2, 2)
    audio = torch.ones(1, 128, 3)
    _, audio_output = adapter(
        [video, audio], torch.tensor([500.0]), context=torch.zeros(1, 2, 4),
        dreamx_audio_scale=2.0, dreamx_video_shift=5.0, dreamx_audio_shift=2.5,
    )
    sigma_a = time_shift_sigma(torch.tensor([0.5]), 5.0, 2.5)
    audio_call = joint.last[1]
    torch.testing.assert_close(audio_call["t"], sigma_a * 1000.0)
    torch.testing.assert_close(torch.stack(audio_call["x"]), audio * (sigma_a / 0.5))
    expected = -(audio * (sigma_a / 0.5)) + (1.0 + sigma_a) * (audio * (sigma_a / 0.5) * 3.0)
    torch.testing.assert_close(audio_output, expected)


def test_nested_latent_roundtrip():
    model = DreamXCreatorBaseModel(ToyJoint(), torch.float32)
    nested = NestedTensor([
        torch.randn(1, 48, 2, 4, 4),
        torch.randn(1, 128, 7),
    ])
    restored = model.process_latent_out(model.process_latent_in(nested))
    for actual, expected in zip(restored.unbind(), nested.unbind()):
        torch.testing.assert_close(actual, expected)


def test_inpaint_scaling_accepts_current_comfy_sampler_keywords():
    model = DreamXCreatorBaseModel(ToyJoint(), torch.float32)
    latent = torch.randn(1, 1, 8)
    result = model.scale_latent_inpaint(
        sigma=torch.ones(1), noise=torch.randn_like(latent), latent_image=latent,
        x=torch.randn_like(latent), denoise_mask=torch.ones_like(latent),
    )
    assert result is latent
