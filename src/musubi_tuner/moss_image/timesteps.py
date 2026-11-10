"""Optional MOSS schedules, separate from the shared Musubi samplers."""
import torch
import math
from functools import lru_cache


def clean_end_policy(spec, epoch):
    """Fizgig's fixed / split / slide / refresh clean-end share, 1-based epochs.

    Adapted from Fizgig fe1fb5d moss_image/driver.py, Apache-2.0.
    The refresh epoch uses half the optimizer LR, not half the gradients.
    """
    if not str(spec or "").strip():
        return None, 1.0
    try:
        parts = [float(p.strip().rstrip("%")) for p in str(spec).split(":")]
    except ValueError as exc:
        raise ValueError("Clean-end share: use a percentage (40), a split (30:7:60), or a slide (30:7:60:8:3).") from exc
    if len(parts) not in (1, 3, 4, 5) or not all(math.isfinite(v) for v in parts):
        raise ValueError("Clean-end share needs 1, 3, 4 or 5 finite numbers.")
    if not 0 <= parts[0] <= 100 or (len(parts) > 1 and not 0 <= parts[2] <= 100):
        raise ValueError("Clean-end percentages must be between 0 and 100.")
    if len(parts) == 1:
        return parts[0], 1.0
    first, until, last = parts[:3]
    end = parts[3] if len(parts) >= 4 else until
    if until < 0 or until != int(until) or end < until or end != int(end):
        raise ValueError("Clean-end epochs must be whole numbers, with slide end >= first epoch.")
    every = parts[4] if len(parts) == 5 else 0
    if every < 0 or every != int(every):
        raise ValueError("Clean-end refresh interval must be a non-negative whole number.")
    share = first if epoch <= until else last if epoch >= end else first + (last - first) * (epoch - until) / (end - until)
    refresh = every > 0 and epoch > end and (epoch - end) % every == 0
    if every > 0 and epoch > end:
        share = min(first, last) if refresh else max(first, last)
    return round(share, 2), (0.5 if refresh and first != last else 1.0)


@lru_cache(maxsize=256)
def clean_share_mean(percent, std=0.95, cutoff=0.95, uniform=0.01):
    """Move the logit-normal centre so P(sigma < 0.5) matches the requested share."""
    want = min(max((percent / 100 - uniform * 0.5) / (1 - uniform), 1e-4), 1 - 1e-4)
    def cdf(x, mean):
        return 1.0 if x == 1 else 0.5 * math.erfc((math.log((1 - x) / x) - mean) / (std * math.sqrt(2)))
    low, high = -12.0, 12.0
    for _ in range(80):
        middle = (low + high) / 2
        if cdf(0.5 * cutoff, middle) / max(cdf(cutoff, middle), 1e-300) < want:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def sigmoid_high_timesteps(count, device, uniform=None):
    """AI Toolkit's sigmoid pool + cubic High/content index distribution.

    Independent adaptation of Ostris AI Toolkit's CustomFlowMatch scheduler
    and BaseSDTrainProcess: redraw a sorted pool of 1000 timesteps per batch,
    then bias its indices toward the high-noise end. Supplied uniform draws
    retain Musubi's optional stratified timestep buckets. RNG draw order across
    trainers is different; this matches the distribution, not seed-for-seed output.

    Reference: ostris/ai-toolkit, commit beb94824c51480a04d3c86c718603676fdef6938,
    toolkit/samplers/custom_flowmatch_sampler.py and jobs/process/BaseSDTrainProcess.py.
    """
    pool = ((1 - torch.sigmoid(torch.randn(1000, device=device))) * 1000).sort(descending=True).values
    u = torch.rand(count, device=device) if uniform is None else torch.as_tensor(uniform, device=device, dtype=torch.float32)
    indices = (u.pow(3) * 1000).long().clamp(0, 999)
    return pool[indices]
