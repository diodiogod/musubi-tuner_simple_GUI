"""Shared GUI controls for weighting still-image diffusion loss with sidecar masks."""

import math

from backends.numeric_settings import decimal_text


MODES = ("Krea 2", "Flux.2 Klein", "Flux.2 Dev", "MiniMax H3 (Experimental)",
         "MiniMax H3 Image (Experimental)", "Wan 2.2")
DEFAULTS = {"image_masked_loss": False, "image_mask_background_weight": "0.1"}
LABELS = {"image_masked_loss": "Use still-image training masks",
          "image_mask_background_weight": "Background weight outside the mask"}
HELP = {
    "image_masked_loss": ("Use matching photo-masklabel.png files beside still training photos. "
                          "The model sees the whole photo, and white areas contribute fully to the diffusion loss. "
                          "Photos without a mask and video clips use the whole frame. Current caches can be reused."),
    "image_mask_background_weight": ("Start with 0.1 to train lightly on the background. Use 0 to ignore black areas "
                                     "or 1 to weight all areas equally. Gray blends between black and white."),
}


def supported(settings):
    mode = settings.get("training_mode")
    return mode in MODES and (
        mode != "MiniMax H3 (Experimental)" or not str(settings.get("minimax_h3_training_workflow") or "").startswith("Video")
    )


def background_weight(settings):
    label = LABELS["image_mask_background_weight"]
    try:
        value = float(decimal_text(settings.get("image_mask_background_weight", DEFAULTS["image_mask_background_weight"])))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number between 0 and 1.") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be a number between 0 and 1.")
    return value
