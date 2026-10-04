"""Offline-first processor/config loading, inspired by Fizgig 06907aaa."""

import os

from huggingface_hub import snapshot_download


def from_pretrained_cache_first(cls, source, **kwargs):
    """Use a cached snapshot directory to avoid Transformers' extra Hub probes.

    Preserve explicit revisions, cache locations and offline requests. An incomplete
    snapshot falls back to the normal loader so first-time downloads still work.
    """
    if os.path.isdir(source):
        return cls.from_pretrained(source, **kwargs)
    snapshot_kwargs = {k: kwargs[k] for k in ("revision", "cache_dir", "token") if k in kwargs}
    if not kwargs.get("force_download", False):
        try:
            local = snapshot_download(source, local_files_only=True, **snapshot_kwargs)
            local_kwargs = dict(kwargs, local_files_only=True)
            return cls.from_pretrained(local, **local_kwargs)
        except (OSError, ValueError):
            pass
    return cls.from_pretrained(source, **kwargs)
