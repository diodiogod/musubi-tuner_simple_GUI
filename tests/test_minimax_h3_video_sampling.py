from types import SimpleNamespace

import torch

from musubi_tuner.minimax_h3.video_sampling import (
    build_joint_sigma_schedule,
    decode_video_latent,
    initialize_joint_noise,
    sample_video_latent,
    write_silent_video,
)


def test_joint_schedule_has_expected_endpoints():
    video, audio = build_joint_sigma_schedule(4, video_shift=12, audio_shift=3)

    assert video.shape == audio.shape == (5,)
    assert video[0] == audio[0] == 1
    assert video[-1] == audio[-1] == 0
    assert torch.all(video[:-1] > video[1:])


def test_initialize_joint_noise_uses_native_22_frame_geometry():
    video, audio = initialize_joint_noise(frame_count=22, width=768, height=512, seed=7, device="cpu")

    assert video.shape == (1, 24, 7, 32, 48)
    assert audio.shape == (1, 32, 2, 37)
    assert video.dtype == torch.float32
    assert audio.dtype == torch.float32


def test_sample_video_latent_runs_joint_layout_on_cpu():
    class FakeTransformer:
        def __call__(self, **kwargs):
            assert kwargs["layout"].task == "t2va"
            assert kwargs["text_token_tags"].eq(1).all()
            return SimpleNamespace(
                video=torch.zeros_like(kwargs["video_latents"]),
                audio=torch.zeros_like(kwargs["audio_latents"]),
            )

    expected, _ = initialize_joint_noise(frame_count=5, width=32, height=32, seed=9, device="cpu")
    result = sample_video_latent(
        FakeTransformer(),
        torch.zeros(3, 5120),
        frame_count=5,
        width=32,
        height=32,
        steps=2,
        seed=9,
        device="cpu",
    )

    assert result.shape == (1, 24, 2, 2, 2)
    assert torch.equal(result, expected)


def test_decode_and_write_silent_mp4(tmp_path):
    class FakeVae:
        def decode(self, latent):
            return torch.zeros((1, 3, 5, 32, 32), dtype=torch.float32)

    pixels = decode_video_latent(FakeVae(), torch.zeros(1, 24, 1, 2, 2), frame_count=5)
    output = write_silent_video(pixels, tmp_path / "preview.mp4", fps=24)

    assert pixels.shape == (5, 32, 32, 3)
    assert output.stat().st_size > 0
