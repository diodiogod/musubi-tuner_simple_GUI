"""CUDA-independent conservative residency budgeting for frozen-weight streaming.

Profiles are process-local: never restore device memory assumptions from a recipe.
The caller must measure *all* backward/auxiliary/optimizer work, excluding previews.
This module does not move tensors or retry failed training steps.
"""

from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True)
class SwapBudget:
    minimum: int = 2
    maximum: int = 48
    reserve_bytes: int = 2 * 1024**3
    observations_before_promotion: int = 2
    promotion_limit: int = 2

    def __post_init__(self):
        if not 2 <= self.minimum <= self.maximum:
            raise ValueError("Automatic swapping requires 2 <= minimum <= maximum")
        if self.reserve_bytes < 0:
            raise ValueError("VRAM reserve cannot be negative")
        if self.observations_before_promotion < 1 or self.promotion_limit < 1:
            raise ValueError("Promotion limits must be positive")


@dataclass
class WorkloadProfile:
    additional_peak: int = 0
    observations: int = 0
    last_streamed: int | None = None


@dataclass(frozen=True)
class SwapDecision:
    streamed: int
    reason: str


class AutomaticSwapPolicy:
    """Choose a count from measured incremental peaks and current headroom.

    ``resident_bytes[s]`` is the dedicated frozen allocation at swap count s;
    ring storage and retained scales belong to the non-evictable baseline.
    ``reclaimable_cache`` must exclude live allocations (reserved - allocated),
    so allocator cache is not counted twice alongside driver free memory.
    """

    def __init__(self, config: SwapBudget, resident_bytes: dict[int, int]):
        self.config = config
        self.resident_bytes = dict(resident_bytes)
        counts = range(config.minimum, config.maximum + 1)
        if any(s not in resident_bytes or resident_bytes[s] < 0 for s in counts):
            raise ValueError("Missing or negative resident byte measurement")
        if any(resident_bytes[s] < resident_bytes[s + 1] for s in range(config.minimum, config.maximum)):
            raise ValueError("Resident bytes must decrease as streaming increases")
        self.profiles: dict[Hashable, WorkloadProfile] = {}
        self.optimizer_ready = False

    def observe(self, key: Hashable, *, baseline_allocated: int, peak_allocated: int, optimizer_ready: bool, streamed: int | None = None):
        if baseline_allocated < 0 or peak_allocated < baseline_allocated:
            raise ValueError("Invalid completed-step allocation measurement")
        profile = self.profiles.setdefault(key, WorkloadProfile())
        # Retain an envelope, not an average that forgets expensive quality passes.
        profile.additional_peak = max(profile.additional_peak, peak_allocated - baseline_allocated)
        profile.observations += 1
        if streamed is not None:
            profile.last_streamed = streamed
        self.optimizer_ready = self.optimizer_ready or optimizer_ready

    def choose(self, key: Hashable, *, current: int, driver_free: int, reclaimable_cache: int) -> SwapDecision:
        cfg = self.config
        if current not in self.resident_bytes or driver_free < 0 or reclaimable_cache < 0:
            raise ValueError("Invalid current memory snapshot")
        profile = self.profiles.get(key)
        if profile is None or not self.optimizer_ready:
            return SwapDecision(cfg.maximum, "learning workload / optimizer warm-up")
        # Current live weights are already subtracted from free memory. Add them
        # back exactly once to compare candidate dedicated residency allocations.
        available = driver_free + reclaimable_cache + self.resident_bytes[current]
        working = profile.additional_peak + cfg.reserve_bytes
        target = next((s for s in range(cfg.minimum, cfg.maximum + 1)
                       if self.resident_bytes[s] + working <= available), cfg.maximum)
        if target > current:
            return SwapDecision(target, "working-memory headroom")
        if profile.observations < cfg.observations_before_promotion:
            return SwapDecision(current, "collecting completed-step measurements")
        # A long bucket must not erase a short bucket's successful residency.
        # Rate-limit newly learned residency, not restoration of a proven plan.
        proven = profile.last_streamed if profile.last_streamed is not None else current
        target = max(target, min(current, proven) - cfg.promotion_limit)
        return SwapDecision(target, "measured workload budget")
