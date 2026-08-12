from argparse import Namespace

import torch

from musubi_tuner.minimax_h3_native_train_network import MiniMaxH3NetworkTrainer
from musubi_tuner.training.trainer_base import DiTOutput


def test_audio_only_loss_drops_video_objective():
    trainer = MiniMaxH3NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([3.0]),
        target=torch.tensor([0.0]),
        extra={
            "audio_pred": torch.tensor([2.0]),
            "audio_target": torch.tensor([0.0]),
            "audio_loss_weight": torch.tensor([0.5], dtype=torch.float32),
        },
    )

    loss, logs = trainer.compute_loss(
        Namespace(audio_only=True), output, torch.tensor([0.5]), None, torch.float32, torch.float32, 0
    )

    assert loss.item() == 2.0
    assert logs["loss/video"].item() == 9.0
    assert logs["loss/video_weight"].item() == 0.0
