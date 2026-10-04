from unittest.mock import Mock, patch

import pytest
import torch

from musubi_tuner.training.preview_recovery import PreviewRecovery
from musubi_tuner.utils.hf_cache import from_pretrained_cache_first
from musubi_tuner.utils.safetensors_utils import TensorWeightAdapter, WeightTransformHooks
from musubi_tuner.modules.host_memory import has_pinning_headroom


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_split_cache_consumed_once(order):
    source = Mock()
    source.keys.return_value = ["fused"]
    source.get_tensor.return_value = torch.arange(8)
    adapter = TensorWeightAdapter(
        WeightTransformHooks(split_hook=lambda k, t: (["a", "b"], t.chunk(2) if t is not None else None)), source
    )
    result = {k: adapter.get_tensor(k) for k in order}
    assert torch.equal(torch.cat([result["a"], result["b"]]), torch.arange(8))
    assert source.get_tensor.call_count == 1
    assert adapter.tensor_cache == {}


@pytest.mark.parametrize("fail_prepare", [False, True])
def test_preview_oom_restores_and_disables(fail_prepare):
    recovery = PreviewRecovery()
    events = []
    rng = torch.get_rng_state().clone()

    def fail():
        torch.manual_seed(123)
        raise torch.OutOfMemoryError("synthetic preview OOM")

    assert not recovery.run(
        fail if fail_prepare else lambda: events.append("prepare"),
        fail, lambda: events.append("restore"), device=torch.device("cpu")
    )
    assert events[-1] == "restore"
    assert torch.equal(rng, torch.get_rng_state())
    assert recovery.disabled
    callback = Mock()
    assert not recovery.run(callback, callback, callback, device=torch.device("cpu"))
    callback.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("bad state"), torch.OutOfMemoryError("restore OOM")])
def test_preview_restoration_failure_is_fatal(error):
    recovery = PreviewRecovery()
    with pytest.raises(type(error)):
        recovery.run(Mock(), Mock(), Mock(side_effect=error), device=torch.device("cpu"))
    assert not recovery.disabled


def test_preview_other_errors_and_distributed_oom_propagate():
    for error, single in [(ValueError("invalid input"), True), (torch.OutOfMemoryError(), False)]:
        restore = Mock()
        with pytest.raises(type(error)):
            PreviewRecovery().run(Mock(), Mock(side_effect=error), restore,
                                  device=torch.device("cpu"), single_process=single)
        restore.assert_called_once()


def test_cached_snapshot_preserves_revision_and_subfolder(tmp_path):
    loader = Mock()
    with patch("musubi_tuner.utils.hf_cache.snapshot_download", return_value=str(tmp_path)) as snapshot:
        from_pretrained_cache_first(loader, "org/repo", revision="abc", subfolder="processor")
    snapshot.assert_called_once_with("org/repo", local_files_only=True, revision="abc")
    loader.from_pretrained.assert_called_once_with(str(tmp_path), revision="abc", subfolder="processor", local_files_only=True)


def test_missing_snapshot_respects_offline():
    loader = Mock()
    with patch("musubi_tuner.utils.hf_cache.snapshot_download", side_effect=OSError("missing")):
        from_pretrained_cache_first(loader, "org/repo", local_files_only=True)
    loader.from_pretrained.assert_called_once_with("org/repo", local_files_only=True)


def test_pinning_headroom():
    with patch("psutil.virtual_memory", return_value=Mock(available=5_000_000_000)):
        assert has_pinning_headroom(500_000_000)
        assert not has_pinning_headroom(4_000_000_000)


def test_staging_pin_failure_falls_back_without_retrying():
    from musubi_tuner.modules.custom_offloading_utils import _StagedCopier

    copier = _StagedCopier.__new__(_StagedCopier)
    copier._staging = None
    copier.num_staging = 2
    copier.device = torch.device("cuda:0")
    with patch("musubi_tuner.modules.host_memory.has_pinning_headroom", return_value=True), \
         patch("torch.empty", return_value=Mock(pin_memory=Mock(side_effect=RuntimeError("pin failed")))) as empty:
        copier._ensure_staging(100)
        copier._ensure_staging(100)
    assert copier._staging == []
    assert empty.call_count == 1


def test_incomplete_snapshot_falls_back_to_requested_repo(tmp_path):
    loader = Mock()
    loader.from_pretrained.side_effect = [OSError("missing tokenizer"), "loaded"]
    with patch("musubi_tuner.utils.hf_cache.snapshot_download", return_value=str(tmp_path)):
        assert from_pretrained_cache_first(loader, "org/repo", revision="abc") == "loaded"
    assert loader.from_pretrained.call_args.args == ("org/repo",)
    assert loader.from_pretrained.call_args.kwargs == {"revision": "abc"}
