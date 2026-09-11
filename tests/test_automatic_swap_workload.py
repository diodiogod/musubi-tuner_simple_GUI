from argparse import Namespace

import torch

from musubi_tuner.training.automatic_swap import batch_geometry, quality_cadence, representative_quality_step, reclaimable_device_cache


def test_allocator_cache_cannot_exceed_physical_headroom():
    assert reclaimable_device_cache(4, 24, 8, 16) == 8
    assert reclaimable_device_cache(0, 24, 8, 30) == 16
    assert reclaimable_device_cache(0, 24, 26, 30) == 0
    assert reclaimable_device_cache(4, 24, 8, 8) == 0


def test_geometry_separates_variable_video_text_and_reference():
    a = {"latents": torch.empty(1, 24, 1, 16, 16), "text": [torch.empty(32, 8)]}
    b = {**a, "text": [torch.empty(64, 8)]}
    c = {**a, "latents": torch.empty(1, 24, 7, 16, 16)}
    d = {**a, "reference": torch.empty(1, 24, 1, 16, 16)}
    assert len({batch_geometry(x) for x in (a, b, c, d)}) == 4


def test_quality_cadence_does_not_reuse_cheap_profile_for_auxiliary_step():
    args = Namespace(h3_dynamic_sigma_every_n_steps=5, depth_anchor_every_n_steps=3,
                     save_every_n_steps=7, sample_every_n_steps=13)
    assert quality_cadence(args, 1) == quality_cadence(args, 2)
    assert quality_cadence(args, 1) != quality_cadence(args, 5)
    assert quality_cadence(args, 3) != quality_cadence(args, 5)
    assert quality_cadence(args, 0) == quality_cadence(args, 15)


def test_random_sigma_skips_cannot_establish_quality_headroom():
    args = Namespace(h3_dynamic_sigma_every_n_steps=5)
    assert not representative_quality_step(args, 5, {"guidance/applied": 0})
    assert representative_quality_step(args, 1, {"guidance/applied": 0})
    assert representative_quality_step(args, 5, {"guidance/applied": 1})
    assert not representative_quality_step(args, 5, {"teacher/conditioned": 0})
