from types import SimpleNamespace

import pytest
import torch

from backends.minimax_h3_face import build_command
from musubi_tuner.face_refinement.draft_minimax_h3 import generate_differentiable_h3, multiframe_reward_loss
from musubi_tuner.face_refinement.h3_frame_policy import validate_frame_policy
from musubi_tuner.minimax_h3.video_sampling import sample_video_latent


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.05))
        self.grad_enabled = []

    def forward(self, **kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        return SimpleNamespace(video=kwargs["video_latents"] * self.weight, audio=kwargs["audio_latents"] * self.weight)

    def forward_image(self, latent, *args):
        self.grad_enabled.append(torch.is_grad_enabled())
        return latent * self.weight


class _VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def decode(self, latent):
        return torch.tanh(torch.nn.functional.interpolate(latent[:, :3], size=(5, 32, 32), mode="trilinear"))


@pytest.mark.parametrize("checkpoint", [True, False])
def test_native_five_frames_backpropagate_only_through_final_steps(checkpoint):
    model = _Model()
    pixels = generate_differentiable_h3(
        model,
        _VAE(),
        torch.zeros(2, 5120),
        resolution=32,
        denoise_steps=4,
        draft_k=2,
        seed=9,
        device=torch.device("cpu"),
        checkpoint_vae=checkpoint,
        training_frame_mode="five_frame",
    )
    assert pixels.shape == (5, 3, 32, 32)
    pixels.square().mean().backward()
    assert model.grad_enabled == [False, False, True, True]
    assert torch.isfinite(model.weight.grad) and model.weight.grad.abs() > 0


def test_default_native_preview_stays_without_gradients():
    model = _Model()
    result = sample_video_latent(model, torch.zeros(2, 5120), frame_count=5, width=32, height=32, steps=4, seed=9, device="cpu")
    assert not result.requires_grad and model.grad_enabled == [False] * 4


def test_single_frame_default_is_identical_to_explicit_mode():
    model = _Model()
    kwargs = {"resolution": 32, "denoise_steps": 4, "draft_k": 1, "seed": 9, "device": torch.device("cpu")}
    default = generate_differentiable_h3(model, _VAE(), torch.zeros(2, 5120), **kwargs)
    explicit = generate_differentiable_h3(model, _VAE(), torch.zeros(2, 5120), training_frame_mode="one_frame", **kwargs)
    assert torch.equal(default, explicit)


class _Reward:
    def __call__(self, pixels, prompt):
        assert pixels.min() >= -1 and pixels.max() <= 1
        self.last_metrics = {"face_detected": 1.0, "face_similarity": float(pixels.detach().mean()), "pose_bucket": "frontal"}
        return pixels.mean().unsqueeze(0)


def test_multiframe_reward_mean_and_worst_penalty_have_correct_gradients():
    pixels = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.9]).reshape(5, 1, 1, 1).expand(5, 3, 2, 2).clone().requires_grad_(True)
    reward = _Reward()
    loss, score = multiframe_reward_loss(reward, pixels, "portrait", 0.25)
    expected = 0.75 * (pixels.detach().mean() * 2 - 1) + 0.25 * (0.2 * 2 - 1)
    assert torch.allclose(score, expected)
    loss.backward()
    assert all(frame.abs().sum() > 0 for frame in pixels.grad)
    assert pixels.grad[0].abs().mean() > pixels.grad[-1].abs().mean()
    assert reward.last_metrics["frame_count"] == 5


@pytest.mark.parametrize("mode,weight", [("bad", 0), ("five_frame", float("nan")), ("five_frame", -1), ("five_frame", 1.1)])
def test_invalid_policy_rejected_before_loading_models(mode, weight):
    with pytest.raises(ValueError):
        validate_frame_policy(mode, weight)


def test_backend_keeps_training_frames_separate_from_quality_preview(tmp_path):
    config = {"training_frame_mode": "five_frame", "worst_frame_weight": 0.25, "quality_preview_mode": "one_frame"}
    command = build_command({}, config, "input", "output", "prompts")
    assert command[command.index("--training_frame_mode") + 1] == "five_frame"
    assert command[command.index("--worst_frame_weight") + 1] == "0.25"
    assert command[command.index("--quality_preview_mode") + 1] == "one_frame"


def test_checkpoint_metadata_records_effective_frame_policy(tmp_path):
    from musubi_tuner.minimax_h3_face_refinement import build_parser, save_network

    args = build_parser().parse_args(
        [
            f"--{key}={key}"
            for key in (
                "dit",
                "vae",
                "text_encoder",
                "network_weights",
                "reference_dir",
                "face_model_dir",
                "prompts_json",
                "output",
            )
        ]
    )

    class Network:
        def save_weights(self, path, dtype, metadata):
            self.metadata = metadata

    network = Network()
    args.training_frame_mode = "five_frame"
    args.worst_frame_weight = 0.0
    save_network(network, tmp_path / "output.safetensors", args, 2)
    assert network.metadata["ss_face_refinement_training_frame_count"] == "5"
    assert network.metadata["ss_face_refinement_reward_aggregation"] == "mean"
    args.worst_frame_weight = 0.25
    save_network(network, tmp_path / "output.safetensors", args, 2)
    assert network.metadata["ss_face_refinement_reward_aggregation"] == "mean_with_worst_frame"
    args.training_frame_mode = "one_frame"
    save_network(network, tmp_path / "output.safetensors", args, 2)
    assert network.metadata["ss_face_refinement_training_frame_count"] == "1"
    assert network.metadata["ss_face_refinement_worst_frame_weight"] == "0.0"
