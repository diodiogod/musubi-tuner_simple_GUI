"""H3 microbatch memory profiling; no dependencies on either GUI."""
import logging
import time

import torch

from musubi_tuner.modules.automatic_swap_policy import AutomaticSwapPolicy, SwapBudget

logger = logging.getLogger(__name__)


def reclaimable_device_cache(free, total, allocated, reserved):
    """Never treat virtual/shared-memory overcommit as physical GPU headroom."""
    return min(max(0, reserved - allocated), max(0, total - free - allocated))


def batch_geometry(value):
    """Include variable-length text/audio/reference tensors, not just video size."""
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), str(value.dtype))
    if isinstance(value, dict):
        return tuple((str(k), batch_geometry(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(batch_geometry(v) for v in value)
    if isinstance(value, (int, float, bool)):
        return value
    return type(value).__name__


def quality_cadence(args, step):
    # Separate optional-pass steps from cheap steps before collecting a profile.
    # Configuration is constant within this controller's process lifetime.
    return tuple((name, step % interval == 0) for name, interval in sorted(vars(args).items())
                 if name.endswith("every_n_steps") and isinstance(interval, int) and interval > 0
                 and not name.startswith(("sample", "save", "log")))


def representative_quality_step(args, step, metrics):
    """Do not learn headroom from a randomly skipped expensive quality pass."""
    cadence = max(1, getattr(args, "h3_dynamic_sigma_every_n_steps", 1))
    if step % cadence == 0 and "guidance/applied" in metrics:
        if float(metrics["guidance/applied"]) == 0:
            return False
    if "teacher/conditioned" in metrics and float(metrics["teacher/conditioned"]) == 0:
        return False
    return True


class AutomaticSwapController:
    def __init__(self, offloader, args):
        self.offloader = offloader
        self.args = args
        self.device = offloader.device
        maximum = min(args.auto_swap_max_blocks, offloader.num_blocks)
        budget = SwapBudget(minimum=args.auto_swap_min_blocks, maximum=maximum,
                            reserve_bytes=int(args.auto_swap_reserve_gb * 1024**3))
        block_bytes = offloader._layout[1]
        self.policy = AutomaticSwapPolicy(budget, {
            s: (offloader.num_blocks - s) * block_bytes
            for s in range(budget.minimum, offloader.num_blocks + 1)
        })
        self.logs = {}
        self.started = False

    def begin(self, batch, step):
        offloader = self.offloader
        self.key = (batch_geometry(batch), quality_cadence(self.args, step))
        self.step = step
        self.start_time = time.perf_counter()
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        # Large alternating buckets can retain a fragmented allocator cache.
        # Release unused storage only under physical pressure, at this graph-free
        # boundary, rather than allowing Windows shared-memory spill to hide it.
        cache_threshold = max(self.policy.config.reserve_bytes, 1024**3)
        if free < cache_threshold and reserved - allocated > cache_threshold:
            with torch.cuda.device(self.device):
                offloader.copier.sync()
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            logger.info("Auto memory: released unused allocator cache under physical memory pressure")
        decision = self.policy.choose(self.key, current=offloader.S, driver_free=free,
                                      reclaimable_cache=reclaimable_device_cache(free, total, allocated, reserved))
        previous = offloader.S
        offloader.reconfigure(decision.streamed)
        torch.cuda.synchronize(self.device)
        self.baseline = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        offloader.begin_microbatch()
        self.started = True
        if previous != offloader.S or self.key not in self.policy.profiles:
            logger.info("Auto memory: %s streaming / %s resident; %.1f GiB reserve; %s "
                        "(driver used %.2f GiB, torch allocated %.2f GiB)",
                        offloader.S, offloader.num_blocks - offloader.S,
                        self.args.auto_swap_reserve_gb, decision.reason,
                        (total - free) / 1024**3, allocated / 1024**3)

    def end(self, optimizer_updated, metrics=None):
        if not self.started:
            raise RuntimeError("Automatic memory measurement has no active microbatch")
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        if representative_quality_step(self.args, self.step, metrics or {}):
            self.policy.observe(self.key, baseline_allocated=self.baseline,
                                peak_allocated=max(self.baseline, peak), optimizer_ready=optimizer_updated,
                                streamed=self.offloader.S)
        self.offloader.end_microbatch()
        self.started = False
        self.logs = {
            "memory/auto_streamed_blocks": self.offloader.S,
            "memory/auto_torch_peak_gib": peak / 1024**3,
            "memory/auto_microbatch_seconds": time.perf_counter() - self.start_time,
        }
        logger.info("Auto memory measured: %d streamed; torch peak %.2f GiB; %.2f s/microbatch (including transition)",
                    self.offloader.S, peak / 1024**3, self.logs["memory/auto_microbatch_seconds"])
