"""Conservative pinned-host-memory admission (Fizgig 81419aa3 inspiration)."""


def has_pinning_headroom(nbytes):
    import psutil

    return psutil.virtual_memory().available >= nbytes + max(4_000_000_000, int(nbytes * 0.75))
