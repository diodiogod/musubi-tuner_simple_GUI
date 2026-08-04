"""Experimental MiniMax-H3 LoRA face-identity refinement using DRaFT-K."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import random

import torch
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.face_refinement import REFERENCE_IMPLEMENTATION, REFERENCE_PAPER
from musubi_tuner.face_refinement.draft import reward_loss, save_preview
from musubi_tuner.face_refinement.draft_minimax_h3 import generate_differentiable_h3
from musubi_tuner.face_refinement.face_reward import FaceSimilarityReward
from musubi_tuner.face_refinement.pose import parse_pose_prompt
from musubi_tuner.face_refinement.pose_plan import PoseProgressTracker
from musubi_tuner.minimax_h3.image_text_encoder import DEFAULT_PROCESSOR_ID, load_minimax_h3_te
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.video_sampling import decode_video_latent, sample_video_latent, write_silent_video
from musubi_tuner.minimax_h3.video_vae import load_video_vae
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.utils.device_utils import clean_memory_on_device


logger = logging.getLogger(__name__)


def load_prompts(path: str) -> tuple[dict | list, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompt_values = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    prompts = [str(item).strip() for item in prompt_values if str(item).strip()]
    if not prompts:
        raise ValueError("Face refinement needs at least one prompt")
    return payload, prompts


def cache_prompt_embeddings(args, prompts: list[str], device: torch.device) -> list[torch.Tensor]:
    encoder = load_minimax_h3_te(
        args.text_encoder,
        device=device,
        compute_dtype=torch.float32,
        quantize=True,
        tokenizer_dir=args.tokenizer,
        load_mode=args.text_encoder_load_mode,
    )
    cached = []
    try:
        with torch.no_grad():
            for prompt in prompts:
                cached.append(encoder.encode(prompt)[0].detach().cpu())
    finally:
        del encoder
        gc.collect()
        clean_memory_on_device(device)
    return cached


def configure_trainable_loras(network, attention_only: bool) -> list[torch.nn.Parameter]:
    allowed = ("_attn_qkv_proj", "_attn_out_proj")
    parameters = []
    for module in network.unet_loras:
        trainable = not attention_only or any(token in module.lora_name for token in allowed)
        module.requires_grad_(trainable)
        if trainable:
            parameters.extend(module.parameters())
    if not parameters:
        raise RuntimeError("No trainable MiniMax-H3 LoRA parameters matched the face-refinement target")
    return parameters


def save_network(network, output: Path, args, step: int) -> None:
    metadata = {
        "ss_training_type": "minimax_h3_draft_face_refinement",
        "ss_minimax_h3_training_mode": "experimental_image_only",
        "ss_face_refinement_steps": str(step),
        "ss_face_refinement_resolution": str(args.resolution),
        "ss_face_refinement_denoise_steps": str(args.denoise_steps),
        "ss_face_refinement_draft_k": str(args.draft_k),
        "ss_face_refinement_pose_aware": str(bool(args.pose_aware)),
        "ss_face_refinement_pose_weight": str(args.pose_reward_weight if args.pose_aware else 0.0),
        "ss_face_refinement_pose_plan": getattr(args, "pose_plan_metadata", ""),
        "ss_face_refinement_quality_preview_mode": args.quality_preview_mode,
        "ss_face_refinement_quality_preview_steps": str(args.quality_preview_steps),
        "ss_face_refinement_reference": REFERENCE_IMPLEMENTATION,
        "ss_face_refinement_paper": REFERENCE_PAPER,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    network.save_weights(str(output), torch.float32, metadata)


def _prepare_pose(args, payload, prompts):
    pose_plan = payload.get("pose_plan") if isinstance(payload, dict) else None
    records = payload.get("prompt_records", []) if isinstance(payload, dict) else []
    tracker = PoseProgressTracker(pose_plan) if args.pose_aware and pose_plan and pose_plan.get("enabled") else None
    if tracker:
        metadata_plan = {
            "preset": pose_plan.get("preset"),
            "overall_anchor_weight": pose_plan.get("overall_anchor_weight"),
            "buckets": {
                pose: {key: value for key, value in config.items() if key != "prompts"}
                for pose, config in pose_plan.get("buckets", {}).items()
            },
        }
        args.pose_plan_metadata = json.dumps(metadata_plan, separators=(",", ":"))
    embedding_prompts = [parse_pose_prompt(prompt)[1] for prompt in prompts] if args.pose_aware else prompts
    weights = [float(item.get("weight", 1.0)) for item in records] if len(records) == len(prompts) else None
    return pose_plan, tracker, embedding_prompts, weights


@torch.no_grad()
def save_five_frame_quality_preview(model, vae, embedding, output_dir: Path, step: int, args) -> tuple[Path, Path]:
    """Render an optional native five-frame evaluation preview without changing the refinement gradient."""
    was_training = model.training
    model.eval()
    try:
        latent = sample_video_latent(
            model,
            embedding,
            frame_count=5,
            width=args.resolution,
            height=args.resolution,
            steps=args.quality_preview_steps,
            seed=args.seed + step,
            device=torch.device(args.device or "cuda"),
        )
        frames = decode_video_latent(vae, latent.to(device=vae.latents_mean.device, dtype=torch.float16), frame_count=5)
    finally:
        model.train(was_training)
    video_path = output_dir / f"face_preview_{step:06d}_five_frames.mp4"
    center_path = output_dir / f"face_preview_{step:06d}_center.png"
    write_silent_video(frames, video_path, fps=24)
    Image.fromarray(frames[2].numpy()).save(center_path)
    return video_path, center_path


def train(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 face refinement requires an NVIDIA CUDA GPU")
    if args.resolution % 32:
        raise ValueError("MiniMax-H3 face-refinement resolution must be divisible by 32")
    if not 1 <= args.draft_k <= args.denoise_steps:
        raise ValueError("draft_k must be between 1 and denoise_steps")
    if not 1 <= args.blocks_to_swap <= 48:
        raise ValueError("MiniMax-H3 face refinement requires 1..48 swapped blocks")
    if args.preview_every < 0 or args.quality_preview_steps < 1:
        raise ValueError("Preview cadence must be zero or greater and quality-preview steps must be positive")
    if args.blocks_to_swap < 30:
        logger.warning("Fewer than 30 swapped blocks is unlikely to fit MiniMax-H3 DRaFT refinement on 24 GB")

    device = torch.device(args.device or "cuda")
    payload, prompts = load_prompts(args.prompts_json)
    pose_plan, pose_tracker, embedding_prompts, prompt_weights = _prepare_pose(args, payload, prompts)
    logger.info("Caching %d compact MiniMax-H3 prompt embedding(s)", len(prompts))
    embeddings = cache_prompt_embeddings(args, embedding_prompts, device)

    logger.info("Loading MiniMax-H3 video VAE on CPU")
    vae = load_video_vae(args.vae, device="cpu", dtype=torch.float16, disable_mmap=args.disable_mmap)
    vae.eval().requires_grad_(False)
    logger.info("Loading pruned ConvRot INT8 MiniMax-H3 transformer")
    model = load_h3_transformer(
        args.dit,
        device="cpu",
        dtype=torch.bfloat16,
        attn_mode=args.attn_mode,
        split_attn=args.split_attn,
        disable_mmap=args.disable_mmap,
        convrot_bwd_mode=args.convrot_bwd_mode,
    ).requires_grad_(False)
    if not model._has_convrot_int8:
        raise ValueError("MiniMax-H3 face refinement requires the pruned ConvRot INT8 transformer")
    model.enable_gradient_checkpointing()
    model.enable_block_swap(
        args.blocks_to_swap,
        BlockSwapConfig(
            device=device,
            supports_backward=True,
            use_pinned_memory=args.use_pinned_memory_for_block_swap,
            h2d_only=True,
            ring_size=args.block_swap_ring_size,
        ),
    )
    model.move_to_device_except_swap_blocks(device)

    weights = load_file(args.network_weights, device="cpu")
    network = lora_minimax_h3.create_arch_network_from_weights(1.0, weights, unet=model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights, strict=True)
    if info.missing_keys or info.unexpected_keys:
        raise ValueError(
            f"MiniMax-H3 face LoRA mismatch: missing={info.missing_keys[:8]}, unexpected={info.unexpected_keys[:8]}"
        )
    network.to(device=device, dtype=torch.float32)
    parameters = configure_trainable_loras(network, args.attention_only)
    model.switch_block_swap_for_training()
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    vae.to(device=device, dtype=torch.float16)
    reference_images = args.reference_dir
    pose_buckets = {}
    if args.reference_manifest:
        manifest = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
        reference_images = manifest.get("reference_images", manifest) if isinstance(manifest, dict) else manifest
        if reference_images and isinstance(reference_images[0], dict):
            pose_buckets = {str(item["path"]): str(item.get("pose", "uncertain")) for item in reference_images}
            reference_images = [str(item["path"]) for item in reference_images if item.get("enabled", True)]
        if not reference_images:
            raise ValueError("The reference manifest contains no enabled face images")
    pose_weight = args.pose_reward_weight if args.pose_aware else 0.0
    pose_targets = {}
    if pose_tracker:
        pose_weight = min(pose_weight, max(0.0, 1.0 - float(pose_plan.get("overall_anchor_weight", 0.80))))
        pose_targets = {
            pose: float(config.get("target", args.target_similarity))
            for pose, config in pose_plan.get("buckets", {}).items()
        }
    reward = FaceSimilarityReward(
        reference_images=reference_images,
        model_dir=args.face_model_dir,
        target_similarity=args.target_similarity,
        reference_entropy_weight=args.anti_copy_weight,
        expression_diversity_weight=0.0,
        pose_buckets=pose_buckets if args.pose_aware else None,
        pose_reward_weight=pose_weight,
        pose_min_references=args.pose_min_references,
        pose_targets=pose_targets,
        providers=["CPUExecutionProvider"],
        device=device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.parent / "face_refinement_pose_metrics.jsonl"
    randomizer = random.Random(args.seed)
    detected_steps = 0
    consecutive_target = 0
    for step in range(1, args.train_steps + 1):
        prompt_index = (
            randomizer.choices(range(len(prompts)), weights=prompt_weights, k=1)[0]
            if prompt_weights
            else randomizer.randrange(len(prompts))
        )
        prompt = prompts[prompt_index]
        optimizer.zero_grad(set_to_none=True)
        pixels = generate_differentiable_h3(
            model,
            vae,
            embeddings[prompt_index],
            resolution=args.resolution,
            denoise_steps=args.denoise_steps,
            draft_k=args.draft_k,
            seed=args.seed + step,
            device=device,
            checkpoint_vae=args.checkpoint_vae,
        )
        loss, reward_value = reward_loss(reward, pixels, prompt)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        metrics = reward.last_metrics
        detected_steps += int(metrics.get("face_detected", 0))
        similarity = float(metrics.get("face_similarity", 0.0))
        consecutive_target = consecutive_target + 1 if similarity >= args.stop_similarity else 0
        print(
            f"step={step}/{args.train_steps} loss={loss.item():.6f} reward={reward_value.item():.6f} "
            f"face_similarity={similarity:.4f} face_detected={int(metrics.get('face_detected', 0))} "
            f"grad_norm={float(grad_norm):.4f}",
            flush=True,
        )
        if pose_tracker:
            pose_bucket = str(metrics.get("pose_bucket") or parse_pose_prompt(prompt)[0] or "")
            if pose_bucket:
                pose_similarity = float(metrics.get("pose_similarity", 0.0))
                pose_tracker.update(pose_bucket, pose_similarity)
                status = pose_tracker.pose_status(pose_bucket)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "pose": pose_bucket, "similarity": pose_similarity, **status}) + "\n")
        if args.save_every > 0 and step % args.save_every == 0:
            save_network(network, output.with_name(f"{output.stem}-{step:06d}{output.suffix}"), args, step)
        if step >= args.min_steps and detected_steps / step < args.min_detection_rate:
            raise RuntimeError("Face detection rate fell below the configured safety threshold")
        pose_stop_reason = pose_tracker.stop_reason() if pose_tracker else None
        stopping = bool(pose_stop_reason or (not pose_tracker and consecutive_target >= args.early_stop_patience))
        preview_due = bool(
            (args.preview_every > 0 and (step == 1 or step % args.preview_every == 0))
            or (args.quality_preview_final and (step == args.train_steps or stopping))
        )
        if preview_due and args.quality_preview_mode == "one_frame":
            print(f"face_preview={save_preview(pixels, output.parent, step, prompt)}", flush=True)
        if preview_due and args.quality_preview_mode == "five_frame":
            # The training reward above remains the fast differentiable one-frame path. This is an
            # additional no-grad quality check and can materially increase wall time at short cadences.
            del loss, reward_value, pixels
            gc.collect()
            clean_memory_on_device(device)
            video_path, center_path = save_five_frame_quality_preview(
                model, vae, embeddings[prompt_index], output.parent, step, args
            )
            print(f"face_preview={center_path}", flush=True)
            print(f"face_preview_video={video_path}", flush=True)
        if stopping:
            print(f"early_stop={pose_stop_reason or 'target_similarity_reached'}", flush=True)
            save_network(network, output, args, step)
            return
    save_network(network, output, args, args.train_steps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMax-H3 DRaFT face-identity LoRA refinement")
    for name in ("dit", "vae", "text_encoder", "network_weights", "reference_dir", "face_model_dir", "prompts_json", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_PROCESSOR_ID)
    parser.add_argument("--text_encoder_load_mode", choices=("auto", "direct", "nf4"), default="auto")
    parser.add_argument("--reference_manifest")
    parser.add_argument("--pose_aware", action="store_true")
    parser.add_argument("--pose_reward_weight", type=float, default=0.20)
    parser.add_argument("--pose_min_references", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--denoise_steps", type=int, default=12)
    parser.add_argument("--draft_k", type=int, default=1)
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
    parser.add_argument("--quality_preview_mode", choices=("one_frame", "five_frame"), default="one_frame")
    parser.add_argument("--quality_preview_steps", type=int, default=20)
    parser.add_argument("--quality_preview_final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_vae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_mode", choices=("torch", "flash", "sageattn", "xformers", "flash3"), default="torch")
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=35)
    parser.add_argument("--block_swap_ring_size", type=int, default=2)
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument("--convrot_bwd_mode", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--disable_mmap", action="store_true")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(build_parser().parse_args())
