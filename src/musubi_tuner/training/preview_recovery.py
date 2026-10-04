"""Recover optional previews without hiding training or restoration failures.

Inspired by Fizgig 4816478e / 815648b2. Recovery is single-process only:
distributed ranks must not continue independently after a failed collective.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PreviewRecovery:
    def __init__(self):
        self.disabled = False

    def run(self, prepare, render, restore, *, device, single_process=True):
        if self.disabled:
            return False
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        failed_oom = False
        try:
            try:
                prepare()
                render()
            except torch.OutOfMemoryError:
                if not single_process:
                    raise
                failed_oom = True
                logger.warning("Preview ran out of memory; restoring training before continuing.", exc_info=True)
            finally:
                # Restoration errors must propagate: continuing with wrong weights is unsafe.
                restore()
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        if failed_oom:
            self.disabled = True
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logger.warning("Training restored. Further previews are disabled for this run after the preview OOM.")
            return False
        return True
