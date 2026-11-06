"""Standalone MOSS DRaFT face-identity refinement of an existing LoRA."""
import gc
import json
import logging
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file

from musubi_tuner.face_refinement import REFERENCE_IMPLEMENTATION, REFERENCE_PAPER
from musubi_tuner.face_refinement.draft_moss_image import generate_differentiable_moss
from musubi_tuner.face_refinement.pose import parse_pose_prompt
from musubi_tuner.face_refinement.pose_plan import PoseProgressTracker
from musubi_tuner.face_refinement.runner import run_face_loop
from musubi_tuner.krea2_face_refinement import build_parser as shared_parser, load_prompts
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.moss_image.checkpoint import load_model
from musubi_tuner.moss_image.sampling import parse_skip_layers
from musubi_tuner.moss_image.text_encoder import load_text_encoder, load_tokenizer, encode_prompts, pad_text_embeds
from musubi_tuner.moss_image.vae import load_vae
from musubi_tuner.networks import lora_moss_image

logger = logging.getLogger(__name__)


def configure_trainable_loras(network, attention_only):
    parameters = []
    for module in network.unet_loras:
        trainable = not attention_only or module.lora_name.endswith(("_proj_q", "_proj_k", "_proj_v", "_proj_o"))
        module.requires_grad_(trainable)
        if trainable:
            parameters.extend(module.parameters())
    if not parameters:
        raise ValueError("No MOSS attention LoRA layers found in the selected adapter.")
    return parameters


def save_network(network, output, args, step):
    metadata = {
        "ss_training_type": "moss_image_draft_face_refinement", "ss_face_refinement_steps": str(step),
        "ss_face_refinement_resolution": str(args.resolution), "ss_face_refinement_denoise_steps": str(args.denoise_steps),
        "ss_face_refinement_draft_k": str(args.draft_k), "ss_face_refinement_pose_aware": str(bool(args.pose_aware)),
        "ss_face_refinement_pose_weight": str(args.pose_reward_weight if args.pose_aware else 0.0),
        "ss_face_refinement_pose_plan": getattr(args, "pose_plan_metadata", ""),
        "ss_face_refinement_reference": REFERENCE_IMPLEMENTATION, "ss_face_refinement_paper": REFERENCE_PAPER,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    network.save_weights(str(output), torch.float32, metadata)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Face refinement requires an NVIDIA CUDA GPU.")
    if args.resolution < 16 or args.resolution % 16 or not 1 <= args.draft_k <= args.denoise_steps:
        raise ValueError("MOSS refinement needs a multiple-of-16 resolution and 1 <= DRaFT K <= denoising steps.")
    if not 0 <= args.blocks_to_swap <= 40:
        raise ValueError("MOSS refinement supports 0–40 swapped blocks.")
    if args.fp8_scaled:
        raise ValueError("MOSS face refinement supports BF16 or ConvRot INT8, not FP8.")
    index = (max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
             if args.gpu_id == "auto" else int(args.gpu_id))
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"GPU index {index} is unavailable.")
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    logger.info("MOSS face refinement using GPU %d: %s", index, torch.cuda.get_device_name(index))
    payload = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    prompts = load_prompts(args.prompts_json)
    pose_plan = payload.get("pose_plan") if isinstance(payload, dict) else None
    prompt_records = payload.get("prompt_records", []) if isinstance(payload, dict) else []
    pose_tracker = PoseProgressTracker(pose_plan) if args.pose_aware and pose_plan and pose_plan.get("enabled") else None
    if pose_tracker:
        metadata = {"preset": pose_plan.get("preset"), "overall_anchor_weight": pose_plan.get("overall_anchor_weight"),
                    "buckets": {p: {k: v for k, v in cfg.items() if k != "prompts"} for p, cfg in pose_plan.get("buckets", {}).items()}}
        args.pose_plan_metadata = json.dumps(metadata, separators=(",", ":"))
    tokenizer = load_tokenizer(args.tokenizer_cache_dir)
    encoder = load_text_encoder(args.text_encoder, device=device, cpu_streaming=args.text_encoder_cpu_streaming)
    plain = [parse_pose_prompt(prompt)[1] for prompt in prompts] if args.pose_aware else prompts
    embeddings_raw = encode_prompts(encoder, tokenizer, [*plain, ""], args.moss_max_token_length)
    negative, negative_mask = pad_text_embeds([embeddings_raw[-1]])
    embeddings = []
    for text in embeddings_raw[:-1]:
        positive, positive_mask = pad_text_embeds([text])
        embeddings.append((positive, positive_mask, negative, negative_mask))
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    vae = load_vae(args.vae).eval().requires_grad_(False)
    model = load_model(args.dit, device="cpu" if args.blocks_to_swap else device,
                       convrot_int8=args.convrot_int8, convrot_bwd_mode=args.convrot_bwd_mode,
                       attn_mode=args.attn_mode, split_attn=args.split_attn).requires_grad_(False)
    if args.blocks_to_swap:
        config = BlockSwapConfig(device=device, supports_backward=True, use_pinned_memory=False, h2d_only=True)
        model.enable_block_swap(args.blocks_to_swap, config)
        model.move_to_device_except_swap_blocks(device)
    model.enable_gradient_checkpointing()
    model.train()
    network = lora_moss_image.create_arch_network_from_weights(1.0, load_file(args.network_weights), unet=model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.load_weights(args.network_weights)
    network.to(device=device, dtype=torch.float32)
    parameters = configure_trainable_loras(network, args.qkvo_only)
    if args.blocks_to_swap:
        model.switch_block_swap_for_training()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    sampler = partial(generate_differentiable_moss, flow_shift=args.moss_sample_flow_shift,
                      skip_layers=parse_skip_layers(args.moss_sample_skip_layers))
    run_face_loop(args, model, vae, network, parameters, optimizer, embeddings, prompts,
                  prompt_records, pose_plan, pose_tracker, device, sampler, save_network)


def build_parser():
    parser = shared_parser()
    parser.description = "MOSS-Image experimental DRaFT face refinement"
    parser.add_argument("--tokenizer_cache_dir")
    parser.add_argument("--text_encoder_cpu_streaming", action="store_true")
    parser.add_argument("--moss_max_token_length", type=int, default=4096)
    parser.add_argument("--moss_sample_flow_shift", type=float, default=5.0)
    parser.add_argument("--moss_sample_skip_layers", default="22")
    parser.add_argument("--convrot_int8", action="store_true")
    parser.add_argument("--convrot_bwd_mode", choices=("bf16", "int8"), default="bf16")
    parser.set_defaults(cfg_scale=3.0)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(build_parser().parse_args())
