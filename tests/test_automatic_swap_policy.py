import pytest

from musubi_tuner.modules.automatic_swap_policy import AutomaticSwapPolicy, SwapBudget


def policy():
    return AutomaticSwapPolicy(SwapBudget(minimum=2, maximum=8, reserve_bytes=100),
                               {s: (10 - s) * 100 for s in range(2, 9)})


def learn(p, key="short", count=2):
    for _ in range(count):
        p.observe(key, baseline_allocated=1000, peak_allocated=1300, optimizer_ready=True)


def test_unknown_and_optimizer_warmup_are_conservative():
    p = policy()
    assert p.choose("new", current=2, driver_free=9999, reclaimable_cache=0).streamed == 8
    p.observe("new", baseline_allocated=10, peak_allocated=20, optimizer_ready=False)
    assert p.choose("new", current=2, driver_free=9999, reclaimable_cache=0).streamed == 8


def test_promotions_are_bounded_and_need_repeated_observations():
    p = policy()
    learn(p, count=1)
    assert p.choose("short", current=8, driver_free=9999, reclaimable_cache=0).streamed == 8
    learn(p, count=1)
    assert p.choose("short", current=8, driver_free=9999, reclaimable_cache=0).streamed == 6


def test_pressure_demotes_without_rate_limit():
    p = policy()
    learn(p)
    assert p.choose("short", current=2, driver_free=0, reclaimable_cache=0).streamed == 6


def test_allocator_cache_and_current_residency_counted_once():
    p = policy()
    learn(p)
    assert p.choose("short", current=4, driver_free=0, reclaimable_cache=200).streamed == 6


def test_profile_is_incremental_not_absolute_and_retains_expensive_pass():
    p = policy()
    learn(p)
    p.observe("short", baseline_allocated=2000, peak_allocated=2100, optimizer_ready=True)
    assert p.profiles["short"].additional_peak == 300
    assert "long" not in p.profiles


def test_invalid_config_and_layout():
    with pytest.raises(ValueError):
        SwapBudget(minimum=0)
    with pytest.raises(ValueError):
        AutomaticSwapPolicy(SwapBudget(), {})


def test_less_free_memory_never_promotes_more_blocks():
    p = policy()
    learn(p)
    decisions = [p.choose("short", current=4, driver_free=free, reclaimable_cache=0).streamed
                 for free in range(0, 1001, 25)]
    assert decisions == sorted(decisions, reverse=True)


def test_change_to_unknown_long_bucket_demotes_before_execution():
    p = policy()
    learn(p)
    assert p.choose("short", current=4, driver_free=2000, reclaimable_cache=0).streamed == 2
    assert p.choose("long", current=2, driver_free=2000, reclaimable_cache=0).streamed == 8


def test_profiled_larger_peak_reserves_more_working_memory():
    p = policy()
    learn(p, "short")
    for _ in range(2):
        p.observe("long", baseline_allocated=1000, peak_allocated=1700, optimizer_ready=True)
    short = p.choose("short", current=4, driver_free=200, reclaimable_cache=0).streamed
    long = p.choose("long", current=4, driver_free=200, reclaimable_cache=0).streamed
    assert long > short


def test_alternating_long_bucket_does_not_reset_short_bucket_residency_learning():
    p = policy()
    for _ in range(2):
        p.observe("short", baseline_allocated=1000, peak_allocated=1300, optimizer_ready=True, streamed=4)
    assert p.choose("new-long", current=4, driver_free=2000, reclaimable_cache=0).streamed == 8
    assert p.choose("short", current=8, driver_free=2000, reclaimable_cache=0).streamed == 2
