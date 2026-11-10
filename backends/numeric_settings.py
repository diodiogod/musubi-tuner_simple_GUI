"""Locale-independent decimal settings at GUI, recipe and command boundaries."""
import re
import math

DECIMAL_KEYS = {
    "learning_rate", "max_grad_norm", "lr_warmup_steps", "network_alpha_low", "network_alpha_high",
    "moss_text_pad_multiple",  # integer validation remains with the owning feature
}
DECIMAL_SUFFIXES = (
    "_sigma", "_weight", "_decay", "_multiplier", "_shift", "_mean", "_std", "_ratio",
    "_cutoff", "_min_lr", "_max_lr", "_learning_rate", "_every_n_epochs", "_sigma_min", "_sigma_max",
)
DECIMAL_PATTERN = re.compile(r"^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?$")


def decimal_setting(key):
    return key in DECIMAL_KEYS or key.endswith(DECIMAL_SUFFIXES)


def decimal_text(value):
    """Accept either decimal separator, retaining scientific notation and blanks.

    Never interpret thousands separators or touch free-form text such as paths,
    captions, optimizer tuples and structured schedules.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    return text.replace(",", ".") if DECIMAL_PATTERN.fullmatch(text) else value


def normalize_numbers(settings):
    return {key: decimal_text(value) if decimal_setting(key) else value for key, value in settings.items()}


class NumericSettingError(ValueError):
    def __init__(self, key):
        self.field = key
        super().__init__("Enter a valid number using a comma or a dot for decimals, without thousands separators.")


def number_value(settings, key, default=None):
    try:
        value = float(decimal_text(settings.get(key, default)))
        if not math.isfinite(value):
            raise ValueError
        return value
    except (TypeError, ValueError) as exc:
        raise NumericSettingError(key) from exc
