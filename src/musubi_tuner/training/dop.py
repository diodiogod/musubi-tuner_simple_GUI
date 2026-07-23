"""Differential Output Preservation helpers.

The loss is architecture-neutral. Architecture-specific cache scripts only need
to store a second text embedding under the appropriate ``dop_*`` batch key.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import re

import torch
import torch.nn.functional as F
from safetensors import safe_open


DOP_SIGNATURE_KEY = "dop_signature"


def dop_enabled(args) -> bool:
    return float(getattr(args, "dop_loss_weight", 0.0) or 0.0) > 0


def validate_dop_config(trigger_word: str, class_word: str, loss_weight: float | None = None) -> tuple[str, str]:
    trigger = str(trigger_word or "").strip()
    class_name = str(class_word or "").strip()
    if not trigger:
        raise ValueError("DOP needs a trigger word, for example the unique name used in your captions.")
    if not class_name:
        raise ValueError("DOP needs a natural preservation phrase, for example 'a person', 'a woman', 'a man', or 'a dog'.")
    if trigger.casefold() == class_name.casefold():
        raise ValueError("The DOP trigger word and preservation class must be different.")
    if loss_weight is not None and float(loss_weight) < 0:
        raise ValueError("DOP loss weight must be zero or greater.")
    return trigger, class_name


def make_class_caption(caption: str, trigger_word: str, class_word: str) -> str:
    """Replace every literal trigger occurrence, case-insensitively.

    Literal matching intentionally supports multi-word triggers and punctuation.
    A missing trigger is an error: silently regularizing the unchanged subject
    caption would teach a different objective than DOP.
    """

    trigger, class_name = validate_dop_config(trigger_word, class_word)
    result, count = re.subn(re.escape(trigger), class_name, str(caption or ""), flags=re.IGNORECASE)
    if count == 0:
        preview = str(caption or "").strip().replace("\n", " ")[:120]
        raise ValueError(
            f"DOP could not find trigger {trigger!r} in caption {preview!r}. "
            "Every training caption must contain the trigger while DOP is enabled."
        )
    return result


def dop_signature(trigger_word: str, class_word: str) -> torch.Tensor:
    trigger, class_name = validate_dop_config(trigger_word, class_word)
    # Keep exact spelling/capitalization in the signature because the class text
    # itself is encoded and capitalization can change tokenizer output.
    digest = hashlib.sha256(f"dop-v1\0{trigger}\0{class_name}".encode("utf-8")).digest()
    return torch.tensor(list(digest[:16]), dtype=torch.uint8)


def validate_cached_signature(batch: dict, trigger_word: str, class_word: str) -> None:
    cached = batch.get(DOP_SIGNATURE_KEY)
    if cached is None:
        raise ValueError(
            "DOP class-caption embeddings are missing from the text cache. "
            "Enable Re-cache Text Encoder Outputs and run the job again."
        )
    expected = dop_signature(trigger_word, class_word).cpu()
    rows = cached.detach().cpu().reshape(-1, expected.numel())
    if any(not torch.equal(row, expected) for row in rows):
        raise ValueError(
            "The cached DOP captions were made with a different trigger word or preservation class. "
            "Re-cache Text Encoder Outputs for this DOP configuration."
        )


def is_valid_dop_cache(item, trigger_word: str, class_word: str, required_key_prefix: str) -> bool:
    """Cheaply validate one existing dual-caption cache without loading embeddings."""

    path = str(getattr(item, "text_encoder_output_cache_path", "") or "")
    if not path:
        return False
    try:
        expected = dop_signature(trigger_word, class_word).cpu()
        with safe_open(path, framework="pt", device="cpu") as cache:
            metadata = cache.metadata() or {}
            if metadata.get("caption1", "") != str(getattr(item, "caption", "") or ""):
                return False
            keys = list(cache.keys())
            if "dop_signature" not in keys or not any(key.startswith(required_key_prefix) for key in keys):
                return False
            return torch.equal(cache.get_tensor("dop_signature").reshape(-1).cpu(), expected)
    except (OSError, RuntimeError, ValueError):
        return False


@contextmanager
def adapter_multiplier(network, multiplier: float):
    """Temporarily set the adapter multiplier, including LoRA/LoHa/LoKr."""

    if not hasattr(network, "set_multiplier"):
        raise TypeError("DOP requires a network module that supports set_multiplier().")
    previous = float(getattr(network, "multiplier", 1.0))
    network.set_multiplier(multiplier)
    try:
        yield
    finally:
        network.set_multiplier(previous)


def compute_dop_loss(
    trainer,
    args,
    accelerator,
    transformer,
    network,
    batch: dict,
    latents: torch.Tensor,
    noise: torch.Tensor,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    network_dtype: torch.dtype,
    *,
    embedding_key: str,
    dop_embedding_key: str,
) -> tuple[torch.Tensor | None, dict]:
    """Compute DOP separately so its graph need not overlap the primary loss graph."""

    if not dop_enabled(args):
        return None, {}

    trigger, class_name = validate_dop_config(args.dop_trigger_word, args.dop_class_word, args.dop_loss_weight)
    validate_cached_signature(batch, trigger, class_name)
    if dop_embedding_key not in batch:
        raise ValueError(
            "DOP class-caption embeddings are missing from this batch. "
            "Enable Re-cache Text Encoder Outputs and run the job again."
        )

    class_batch = dict(batch)
    class_batch[embedding_key] = batch[dop_embedding_key]
    unwrapped_network = accelerator.unwrap_model(network)

    # Teacher: frozen base model, generic class caption. This output is detached.
    with torch.no_grad(), adapter_multiplier(unwrapped_network, 0.0):
        prior_output = trainer.call_dit(
            args,
            accelerator,
            transformer,
            latents,
            class_batch,
            noise,
            noisy_model_input,
            timesteps,
            network_dtype,
        )
        prior_pred = prior_output.pred.detach()

    # Student: LoRA enabled, same generic class caption. Gradients only update the adapter.
    preservation_output = trainer.call_dit(
        args,
        accelerator,
        transformer,
        latents,
        class_batch,
        noise,
        noisy_model_input,
        timesteps,
        network_dtype,
    )
    preservation_loss = F.mse_loss(preservation_output.pred.float(), prior_pred.float())
    weighted = preservation_loss * float(args.dop_loss_weight)
    return weighted, {
        "loss/dop": preservation_loss.detach(),
        "loss/dop_weighted": weighted.detach(),
    }


def add_dop_loss(
    trainer, args, accelerator, transformer, network, batch, latents, noise,
    noisy_model_input, timesteps, network_dtype, normal_loss, metrics, *,
    embedding_key: str, dop_embedding_key: str,
) -> tuple[torch.Tensor, dict]:
    """Compatibility wrapper for callers that want one combined scalar."""

    weighted, dop_metrics = compute_dop_loss(
        trainer, args, accelerator, transformer, network, batch, latents, noise,
        noisy_model_input, timesteps, network_dtype,
        embedding_key=embedding_key, dop_embedding_key=dop_embedding_key,
    )
    if weighted is None:
        return normal_loss, metrics
    result_metrics = dict(metrics)
    result_metrics.update(dop_metrics)
    return normal_loss + weighted.to(normal_loss.dtype), result_metrics


def add_cache_arguments(parser) -> None:
    group = parser.add_argument_group("Differential Output Preservation text cache")
    group.add_argument("--dop_trigger_word", default="", help="Subject trigger word to replace in every caption.")
    group.add_argument("--dop_class_word", default="", help="Natural class phrase replacing the trigger, such as 'a person' or 'a dog'.")


def add_training_arguments(parser) -> None:
    group = parser.add_argument_group("Differential Output Preservation (experimental)")
    group.add_argument("--dop_loss_weight", type=float, default=0.0, help="DOP preservation strength; 0 disables DOP.")
    group.add_argument("--dop_trigger_word", default="", help="Subject trigger word present in every caption.")
    group.add_argument("--dop_class_word", default="", help="Natural preservation phrase replacing the trigger, such as 'a person'.")
