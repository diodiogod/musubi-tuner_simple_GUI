"""Copy-stream allocator registration must cover both transfer engines (upstream #1150)."""
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
from musubi_tuner.modules.custom_offloading_utils import _DirectCopier, _StagedCopier


@pytest.mark.parametrize("staged", [False, True])
def test_destination_is_registered_on_copy_stream(staged):
    copier = (_StagedCopier if staged else _DirectCopier).__new__(_StagedCopier if staged else _DirectCopier)
    copier.copy_stream = Mock()
    copier.debug = False
    src, dst = Mock(), Mock()
    src.is_pinned.return_value = False
    with patch("torch.cuda.stream", return_value=nullcontext()):
        if staged:
            copier._rr, copier.num_staging = 0, 1
            copier._staging, copier._staging_free = [], [None]
            copier._run(dst, src, None)
        else:
            copier._events = {}
            copier.submit(0, dst, src)
    dst.record_stream.assert_called_once_with(copier.copy_stream)
    dst.copy_.assert_called_once_with(src, non_blocking=False)
