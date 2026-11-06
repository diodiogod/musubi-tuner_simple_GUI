"""Standalone Krea 2 LoRA face-identity refinement using DRaFT-K."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import random

import torch
from safetensors.torch import load_file

from musubi_tuner.face_refinement import REFERENCE_IMPLEMENTATION, REFERENCE_PAPER
from musubi_tuner.face_refinement.draft import generate_differentiable, reward_loss, save_preview
from musubi_tuner.face_refinement.face_reward import FaceSimilarityReward
from musubi_tuner.face_refinement.pose import parse_pose_prompt
from musubi_tuner.face_refinement.pose_plan import PoseProgressTracker
from musubi_tuner.krea2 import krea2_sampling, krea2_utils
from musubi_tuner.networks import lora_krea2
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.qwen_image import qwen_image_utils

logger = logging.getLogger(__name__)


def load_prompts(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    prompts = [str(item).strip() for item in prompts if str(item).strip()]
    if not prompts:
        raise ValueError("Face refinement needs at least one prompt")
    return prompts


def cache_prompt_embeddings(text_encoder_path: str, prompts: list[str], device: torch.device):
    encoder = krea2_utils.load_krea2_text_encoder(text_encoder_path, dtype=torch.bfloat16, device=device)
    cached = []
    with torch.no_grad():
        for prompt in prompts:
            text, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [prompt])
            text, mask = krea2_sampling.gather_valid_text(text, mask)
            negative, negative_mask = krea2_utils.get_krea2_prompt_embeds(encoder, [""])
            negative, negative_mask = krea2_sampling.gather_valid_text(negative, negative_mask)
            cached.append((text.cpu(), mask.cpu(), negative.cpu(), negative_mask.cpu()))
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cached


def configure_trainable_loras(network, qkvo_only: bool) -> list[torch.nn.Parameter]:
    allowed = ("_wq", "_wk", "_wv", "_wo")
    parameters = []
    for module in network.unet_loras:
        trainable = not qkvo_only or any(token in module.lora_name for token in allowed)
        module.requires_grad_(trainable)
        if trainable:
            parameters.extend(module.parameters())
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters matched the face-refinement target")
    return parameters


def save_network(network, output: Path, args, step: int) -> None:
    metadata = {
        "ss_training_type": "krea2_draft_face_refinement",
        "ss_face_refinement_steps": str(step),
        "ss_face_refinement_resolution": str(args.resolution),
        "ss_face_refinement_denoise_steps": str(args.denoise_steps),
        "ss_face_refinement_draft_k": str(args.draft_k),
        "ss_face_refinement_pose_aware": str(bool(args.pose_aware)),
        "ss_face_refinement_pose_weight": str(args.pose_reward_weight if args.pose_aware else 0.0),
        "ss_face_refinement_pose_plan": getattr(args, "pose_plan_metadata", ""),
        "ss_face_refinement_reference": REFERENCE_IMPLEMENTATION,
        "ss_face_refinement_paper": REFERENCE_PAPER,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    network.save_weights(str(output), torch.float32, metadata)


def train(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Face refinement requires an NVIDIA CUDA GPU")
    if args.resolution % 16:
        raise ValueError("Face-refinement resolution must be divisible by 16")
    if not 1 <= args.draft_k <= args.denoise_steps:
        raise ValueError("draft_k must be between 1 and denoise_steps")
    if not 0 <= args.blocks_to_swap <= 26:
        raise ValueError("blocks_to_swap must be between 0 and 26")
    if args.gpu_id == "auto":
        gpu_index = max(range(torch.cuda.device_count()), key=lambda index: torch.cuda.get_device_properties(index).total_memory)
    else:
        gpu_index = int(args.gpu_id)
    if gpu_index >= torch.cuda.device_count():
        raise ValueError(f"GPU index {gpu_index} is not available")
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    logger.info("Using GPU %d: %s", gpu_index, torch.cuda.get_device_name(gpu_index))
    prompt_payload = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    prompts = load_prompts(args.prompts_json)
    pose_plan = prompt_payload.get("pose_plan") if isinstance(prompt_payload, dict) else None
    prompt_records = prompt_payload.get("prompt_records", []) if isinstance(prompt_payload, dict) else []
    pose_tracker = PoseProgressTracker(pose_plan) if args.pose_aware and pose_plan and pose_plan.get("enabled") else None
    if pose_tracker:
        metadata_plan = {"preset": pose_plan.get("preset"), "overall_anchor_weight": pose_plan.get("overall_anchor_weight"), "buckets": {pose: {key: value for key, value in cfg.items() if key != "prompts"} for pose, cfg in pose_plan.get("buckets", {}).items()}}
        args.pose_plan_metadata = json.dumps(metadata_plan, separators=(",", ":"))
    logger.info("Caching %d face-refinement prompt(s)", len(prompts))
    embedding_prompts = [parse_pose_prompt(prompt)[1] for prompt in prompts] if args.pose_aware else prompts
    embeddings = cache_prompt_embeddings(args.text_encoder, embedding_prompts, device)

    vae = qwen_image_utils.load_vae(args.vae, input_channels=3, device="cpu", disable_mmap=True).eval().requires_grad_(False)
    loading_device = "cpu" if args.blocks_to_swap > 0 else device
    model = krea2_utils.load_krea2_dit(
        args.dit, device=device, dtype=torch.bfloat16, fp8_scaled=args.fp8_scaled,
        loading_device=loading_device, attn_mode=args.attn_mode, split_attn=args.split_attn,
        projector_diff_path=args.projector_diff, projector_diff_strength=args.projector_diff_strength,
    ).requires_grad_(False)
    if args.blocks_to_swap > 0:
        # The base transformer is frozen, so the LoRA-specific H2D-only
        # offloader is both lighter and safe with checkpoint recomputation.
        config = BlockSwapConfig(device=device, supports_backward=True, use_pinned_memory=False, h2d_only=True)
        model.enable_block_swap(args.blocks_to_swap, config)
        model.move_to_device_except_swap_blocks(device)
    model.enable_gradient_checkpointing()
    model.train()
    weights = load_file(args.network_weights)
    network = lora_krea2.create_arch_network_from_weights(1.0, weights, unet=model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.load_weights(args.network_weights)
    network.to(device=device, dtype=torch.float32)
    parameters = configure_trainable_loras(network, args.qkvo_only)
    if args.blocks_to_swap > 0:
        model.switch_block_swap_for_training()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    from musubi_tuner.face_refinement.runner import run_face_loop
    run_face_loop(args, model, vae, network, parameters, optimizer, embeddings, prompts,
                  prompt_records, pose_plan, pose_tracker, device, generate_differentiable, save_network)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Krea 2 DRaFT face-identity LoRA refinement")
    for name in ("dit", "vae", "text_encoder", "network_weights", "reference_dir", "face_model_dir", "prompts_json", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--reference_manifest")
    parser.add_argument("--pose_aware", action="store_true")
    parser.add_argument("--pose_reward_weight", type=float, default=0.20)
    parser.add_argument("--pose_min_references", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--denoise_steps", type=int, default=12)
    parser.add_argument("--draft_k", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_similarity", type=float, default=0.45)
    parser.add_argument("--stop_similarity", type=float, default=0.55)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--min_detection_rate", type=float, default=0.25)
    parser.add_argument("--min_steps", type=int, default=8)
    parser.add_argument("--anti_copy_weight", type=float, default=0.02)
    parser.add_argument("--preview_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qkvo_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_vae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--attn_mode", choices=["torch", "flash", "sageattn", "xformers"], default="torch")
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=10)
    parser.add_argument("--gpu_id", default="auto")
    parser.add_argument("--projector_diff", default=None)
    parser.add_argument("--projector_diff_strength", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(build_parser().parse_args())
