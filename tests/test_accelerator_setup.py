from musubi_tuner.training.accelerator_setup import distributed_launch_requested


def test_multiple_physical_gpus_do_not_imply_distributed_launch(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert not distributed_launch_requested()


def test_launcher_rank_environment_enables_distributed_setup(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert distributed_launch_requested()
