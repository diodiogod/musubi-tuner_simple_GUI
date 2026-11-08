"""Optional MOSS schedules, separate from the shared Musubi samplers."""
import torch


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
