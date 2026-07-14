"""Downstream helper for exact positional recovery from Musubi state folders.

Kept outside ``trainer_base.py`` so future upstream imports only need to preserve the
small opt-in call sites marked ``DOWNSTREAM: exact resume`` in the shared trainer.
"""

from __future__ import annotations

import os
import re


def resume_training_position(resume_path, num_update_steps_per_epoch, gradient_accumulation_steps, batches_per_epoch):
    """Recover loop counters encoded by Musubi's epoch/step state-folder names."""
    position = {"known": False, "epoch": 0, "global_step": 0, "batches_to_skip": 0}
    if not resume_path or num_update_steps_per_epoch <= 0:
        return position
    state_name = os.path.basename(str(resume_path).replace("\\", "/").rstrip("/"))
    step_match = re.search(r"-step(\d+)-state$", state_name, re.IGNORECASE)
    epoch_match = re.search(r"-(\d+)-state$", state_name, re.IGNORECASE)
    if step_match:
        global_step = int(step_match.group(1))
        epoch = global_step // num_update_steps_per_epoch
        updates_into_epoch = global_step % num_update_steps_per_epoch
        batches_to_skip = min(batches_per_epoch, updates_into_epoch * max(1, gradient_accumulation_steps))
        return {"known": True, "epoch": epoch, "global_step": global_step, "batches_to_skip": batches_to_skip}
    if epoch_match:
        completed_epochs = int(epoch_match.group(1))
        return {
            "known": True,
            "epoch": completed_epochs,
            "global_step": completed_epochs * num_update_steps_per_epoch,
            "batches_to_skip": 0,
        }
    return position
