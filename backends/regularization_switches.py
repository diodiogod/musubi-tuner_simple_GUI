"""GUI feature switches over the existing zero-strength training CLI contract."""
import math
from .numeric_settings import decimal_text

# switch -> (strength, suggested first value, fields controlled by the switch)
FEATURES = {
    "krea2_weight_noise_enabled": ("krea2_weight_noise_sigma", "0.0125", ("krea2_weight_noise_",)),
    "krea2_depth_anchor_enabled": ("krea2_depth_anchor_weight", "0.01", (
        "krea2_depth_anchor_", "krea2_keep_depth_helpers_on_gpu", "krea2_depth_vae_device",
        "minimax_h3_depth_vae_device", "minimax_h3_keep_depth_vae_on_device", "minimax_h3_depth_every_n_steps")),
    "moss_weight_noise_enabled": ("moss_weight_noise_sigma", "0.0125", ("moss_weight_noise_",)),
    "moss_depth_anchor_enabled": ("moss_depth_anchor_weight", "0.01", (
        "moss_depth_anchor_", "moss_keep_depth_helpers_on_gpu")),
}
LABELS = {key: "Enable structural depth guidance" if "depth" in key else "Enable adapter weight noise"
          for key in FEATURES}
HELP = {key: ("Off completely disables this feature and keeps your saved strength for later. "
              "On uses the strength below. Old recipes without this switch retain their previous behavior.")
        for key in FEATURES}


def _positive(value):
    try:
        return math.isfinite(float(decimal_text(value) or 0)) and float(decimal_text(value) or 0) > 0
    except (TypeError, ValueError):
        return False


def normalize_switches(settings):
    """Migrate only fields present in the recipe; leave unrelated partial updates alone."""
    result = dict(settings)
    for switch, (strength, _, _) in FEATURES.items():
        if switch not in result and strength in result:
            result[switch] = _positive(result[strength])
    return result


def enabled(settings, switch):
    if switch not in settings:
        return _positive(settings.get(FEATURES[switch][0]))
    value = settings[switch]
    return value is True or value == 1 or str(value).lower() == "true"


def controlled_by(key):
    for switch, (_, _, prefixes) in FEATURES.items():
        if key != switch and any(key.startswith(prefix) for prefix in prefixes):
            return switch
    return None


def effective_settings(settings):
    """Mask inactive feature arguments on a copy, preserving the editable recipe."""
    from .numeric_settings import normalize_numbers
    result = normalize_switches(normalize_numbers(settings))
    for switch, (strength, _, _) in FEATURES.items():
        if not enabled(result, switch):
            for key in list(result):
                if controlled_by(key) == switch:
                    result.pop(key)
            result[strength] = "0"
    return result


def toggle_settings(settings, switch, on):
    result = dict(settings)
    strength, first, _ = FEATURES[switch]
    result[switch] = bool(on)
    if on and not _positive(result.get(strength)):
        result[strength] = first
    return result
