from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


def test_epoch_state_starts_at_the_following_epoch():
    position = Krea2NetworkTrainer._resume_training_position(
        "example-000004-state",
        num_update_steps_per_epoch=10,
        gradient_accumulation_steps=1,
        batches_per_epoch=10,
    )

    assert position == {"known": True, "epoch": 4, "global_step": 40, "batches_to_skip": 0}


def test_step_state_skips_only_batches_already_consumed():
    position = Krea2NetworkTrainer._resume_training_position(
        "example-step00000025-state",
        num_update_steps_per_epoch=10,
        gradient_accumulation_steps=2,
        batches_per_epoch=20,
    )

    assert position == {"known": True, "epoch": 2, "global_step": 25, "batches_to_skip": 10}


def test_unnumbered_state_is_not_claimed_as_positional_resume():
    position = Krea2NetworkTrainer._resume_training_position(
        "example-state",
        num_update_steps_per_epoch=10,
        gradient_accumulation_steps=1,
        batches_per_epoch=10,
    )

    assert position == {"known": False, "epoch": 0, "global_step": 0, "batches_to_skip": 0}
