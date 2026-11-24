"""Experimental still-image LoRA training for the pruned MiniMax-H3 ConvRot INT8 base."""

from __future__ import annotations

import argparse
import gc
import logging
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL,
    ARCHITECTURE_MINIMAX_H3_PICTURE, ARCHITECTURE_MINIMAX_H3_PICTURE_FULL,
)
from musubi_tuner.minimax_h3.image_sampling import H3_PICTURE_SIGMAS, decode_image_latent, sample_image_latent
from musubi_tuner.minimax_h3.image_text_encoder import DEFAULT_PROCESSOR_ID, load_minimax_h3_te
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
from musubi_tuner.minimax_h3.video_sampling import decode_video_latent, sample_video_latent
from musubi_tuner.minimax_h3.video_vae import load_video_vae
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.training.dop import compute_dop_loss, dop_enabled, validate_dop_config
from musubi_tuner.training.h3_guidance_protection import (
    base_sigma_from_model_timestep,
    build_guided_target,
    enabled as guidance_protection_enabled,
    validate as validate_guidance_scale,
    validate_sigma_min,
)
from musubi_tuner.training.h3_training_assistant import (
    DEFAULT_ASSISTANT,
    base_preservation_loss,
    load_live_assistant,
    load_live_foundation_lora,
    should_preserve_base,
)
from musubi_tuner.perceptual.depth_devices import resolve_depth_vae_device
from musubi_tuner.utils.device_utils import clean_memory_on_device


logger = logging.getLogger(__name__)


class MiniMaxH3ImageNetworkTrainer(NetworkTrainer):
    """Image-only H3 path: one still image, no audio, frozen INT8 base."""

    image_mask_stride = 16

    def __init__(self) -> None:
        super().__init__()
        self.vae_frame_stride = 1
        self._dop_step_context = None
        self._weight_noise = None
        self._weight_noise_logs = {}
        self._depth_anchor = None
        self._depth_vae_device = None
        self._training_assistant = None
        self._foundation_lora = None

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 12.0
        legacy_method = args.h3_quality_protection_method
        if legacy_method == "dynamic":
            args.h3_guidance_distillation_protection = True
        elif legacy_method == "assistant":
            args.h3_training_assistant_enabled = True
        elif legacy_method == "assistant_preservation":
            args.h3_training_assistant_enabled = True
            args.h3_base_preservation_enabled = True
        elif legacy_method == "off":
            args.h3_guidance_distillation_protection = False
        active_protection = []
        if args.h3_training_assistant_enabled:
            active_protection.append("assistant")
        if args.h3_guidance_distillation_protection:
            active_protection.append("dynamic")
        if args.h3_base_preservation_enabled:
            active_protection.append("base")
        args.h3_quality_protection_method = "+".join(active_protection) or "off"
        if args.mixed_precision != "bf16":
            raise ValueError("Experimental MiniMax-H3 image training requires --mixed_precision bf16")
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("MiniMax-H3 image training loads the pre-quantized ConvRot INT8 base directly; do not use FP8 flags")
        if args.sample_prompts and not args.text_encoder:
            raise ValueError("--text_encoder is required when MiniMax-H3 training previews are enabled")
        if args.sample_prompts and not args.vae:
            raise ValueError("--vae is required when MiniMax-H3 training previews are enabled")
        if args.compile:
            raise ValueError("torch.compile is not enabled for the experimental ConvRot INT8 MiniMax-H3 path")
        if args.blocks_to_swap and not args.block_swap_h2d_only:
            raise ValueError("MiniMax-H3 ConvRot INT8 block swap requires --block_swap_h2d_only")
        if args.blocks_to_swap and not args.gradient_checkpointing:
            raise ValueError("MiniMax-H3 H2D-only block swap requires --gradient_checkpointing")
        if args.blocks_to_swap is None or args.blocks_to_swap <= 0:
            logger.warning(
                "MiniMax-H3 block swap is disabled. The ~21 GB pruned base leaves too little training headroom on a 24 GB GPU."
            )
        elif args.blocks_to_swap > 48:
            raise ValueError("MiniMax-H3 supports at most 48 swapped blocks out of 50")
        if args.weight_noise_sigma < 0:
            raise ValueError("--weight_noise_sigma must be non-negative")
        if args.depth_anchor_weight < 0:
            raise ValueError("--depth_anchor_weight must be non-negative")
        if args.depth_anchor_input_size <= 0 or args.depth_anchor_input_size % 14:
            raise ValueError("--depth_anchor_input_size must be a positive multiple of 14")
        if args.depth_anchor_every_n_steps <= 0:
            raise ValueError("--depth_anchor_every_n_steps must be positive")
        if args.depth_anchor_weight > 0 and not args.vae:
            raise ValueError("--vae is required when MiniMax-H3 depth anchoring is enabled")
        if args.depth_anchor_weight > 0 and args.depth_anchor_vae_device == "training" and (args.blocks_to_swap or 0) < 30:
            logger.warning(
                "MiniMax-H3 depth anchoring holds the DiT graph, VAE decoder, and depth model together. "
                "Fewer than 30 swapped blocks may exceed 24 GB; start with 30 or more."
            )
        if dop_enabled(args):
            validate_dop_config(args.dop_trigger_word, args.dop_class_word, args.dop_loss_weight)
            logger.info(
                "MiniMax-H3 DOP enabled: trigger=%r, preservation class=%r, strength=%g",
                args.dop_trigger_word,
                args.dop_class_word,
                args.dop_loss_weight,
            )
        if guidance_protection_enabled(args):
            validate_guidance_scale(args.h3_guidance_distillation_scale)
            validate_sigma_min(getattr(args, "h3_guidance_distillation_sigma_min", 0.15))
            logger.info(
                "MiniMax-H3 Dynamic Sigma enabled: scale=%g, every %d step(s), base sigma minimum=%g",
                args.h3_guidance_distillation_scale,
                args.h3_dynamic_sigma_every_n_steps,
                getattr(args, "h3_guidance_distillation_sigma_min", 0.15),
            )
        if args.h3_dynamic_sigma_every_n_steps < 1:
            raise ValueError("H3 Dynamic Sigma cadence must be at least 1")
        if args.h3_training_assistant_enabled and not str(args.h3_training_assistant or "").strip():
            raise ValueError("The selected H3 assistant method requires --h3_training_assistant")
        if getattr(args, "h3_foundation_lora", None) and not torch.isfinite(
            torch.tensor(getattr(args, "h3_foundation_lora_multiplier", 1.0))
        ):
            raise ValueError("H3 foundation LoRA strength must be a finite number")
        if args.h3_base_preservation_loss_weight < 0:
            raise ValueError("H3 base-preservation strength must be non-negative")
        if args.h3_base_preservation_every_n_steps < 1:
            raise ValueError("H3 base-preservation cadence must be at least 1")
        if args.h3_base_preservation_enabled and args.h3_base_preservation_loss_weight <= 0:
            raise ValueError("Enabled H3 base preservation requires a strength greater than zero")
        if (
            args.h3_base_preservation_enabled
            and args.h3_base_preservation_reference == "assistant"
            and not args.h3_training_assistant_enabled
        ):
            raise ValueError("Base + assistant reference requires the Ostris training assistant to be enabled")
        if getattr(args, "h3_teacher_matching", False):
            if args.h3_teacher_conditions not in {"ref", "subject_ref"}:
                raise ValueError(
                    "Compact ConvRot teacher matching supports same-item ('ref') and explicit other-picture "
                    "('subject_ref') teachers. Endpoint teachers require the native video workflow."
                )
            if not 0.0 <= args.h3_teacher_condition_sigma_max <= 1.0:
                raise ValueError("H3 teacher condition sigma maximum must be between 0 and 1")
            if args.h3_teacher_direct_loss_weight < 0:
                raise ValueError("H3 teacher direct loss weight must be non-negative")
            if guidance_protection_enabled(args):
                raise ValueError("Compact teacher matching and Dynamic Sigma are alternative target methods")

    def on_transformer_loaded(self, args, accelerator, transformer) -> None:
        if getattr(args, "h3_foundation_lora", None):
            logger.info(
                "Loading frozen MiniMax-H3 foundation LoRA: %s at %g",
                args.h3_foundation_lora,
                getattr(args, "h3_foundation_lora_multiplier", 1.0),
            )
            self._foundation_lora = load_live_foundation_lora(
                transformer,
                args.h3_foundation_lora,
                getattr(args, "h3_foundation_lora_multiplier", 1.0),
                accelerator.device,
                torch.bfloat16,
            )
            logger.info("MiniMax-H3 foundation LoRA active from %s", self._foundation_lora.foundation_source)
        if args.h3_training_assistant_enabled:
            logger.info("Loading frozen MiniMax-H3 training assistant: %s", args.h3_training_assistant)
            self._training_assistant = load_live_assistant(
                transformer,
                args.h3_training_assistant,
                accelerator.device,
                torch.bfloat16,
            )
            logger.info("MiniMax-H3 training assistant active from %s", self._training_assistant.assistant_source)

    def on_before_sample_images(self, *args, **kwargs) -> None:
        if self._training_assistant is not None:
            self._training_assistant.set_enabled(False)

    def on_after_sample_images(self, *args, **kwargs) -> None:
        if self._training_assistant is not None:
            self._training_assistant.set_enabled(True)

    def _build_dataset(self, args):
        group, collator, current_epoch = super()._build_dataset(args)
        for index, dataset in enumerate(group.datasets):
            if int(dataset.batch_manager.batch_size) != 1:
                raise ValueError(
                    f"MiniMax-H3 image dataset {index} requires batch_size=1; use gradient accumulation for larger batches"
                )
        return group, collator, current_epoch

    def _get_depth_vae_device(self, args, training_device: torch.device) -> torch.device:
        if self._depth_vae_device is None:
            self._depth_vae_device = resolve_depth_vae_device(args.depth_anchor_vae_device, training_device)
            logger.info(
                "MiniMax-H3 depth devices: DiT/depth model=%s, differentiable VAE=%s",
                training_device,
                self._depth_vae_device,
            )
        return self._depth_vae_device

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        prompts = load_prompts(sample_prompts)
        if not prompts:
            raise ValueError("MiniMax-H3 sample prompt file is empty")
        for parameter in prompts:
            if parameter.get("negative_prompt") not in {None, ""}:
                raise ValueError("MiniMax-H3 is guidance-distilled; preview negative prompts and CFG are unsupported")
            picture_model = self.architecture == ARCHITECTURE_MINIMAX_H3_PICTURE
            if picture_model:
                parameter.update(sample_steps=6, frame_count=1)
            parameter.setdefault("sample_steps", 28)
            parameter.setdefault("width", 768)
            parameter.setdefault("height", 768)
            parameter.setdefault("frame_count", 1)
            requested_frames = int(parameter["frame_count"])
            if requested_frames not in {1, 5}:
                logger.warning(
                    "MiniMax-H3 scheduled training preview requested unsupported length %s; using one frame. "
                    "Scheduled previews support 1 or 5 frames; use standalone Preview for 22/39-frame video.",
                    requested_frames,
                )
                parameter["frame_count"] = 1
            parameter.setdefault("guidance_scale", 1.0)
            parameter.setdefault("cfg_scale", 1.0)

        logger.info("Loading compact MiniMax-H3 text encoder for %s training preview prompt(s)", len(prompts))
        encoder = load_minimax_h3_te(
            args.text_encoder,
            device=accelerator.device,
            compute_dtype=torch.float32,
            quantize=True,
            tokenizer_dir=args.tokenizer,
            load_mode=args.text_encoder_load_mode,
            blocks_to_swap=args.text_encoder_blocks_to_swap,
        )
        encoded: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for parameter in prompts:
                    prompt = parameter.get("prompt", "")
                    if prompt not in encoded:
                        options = {"max_length": 4096} if picture_model else {}
                        encoded[prompt] = encoder.encode(prompt, **options)[0].detach().to("cpu")
                    parameter["mmh3_hidden_states"] = encoded[prompt]
        finally:
            del encoder
            gc.collect()
            clean_memory_on_device(accelerator.device)
        return prompts

    def _prepare_sampling(self, args, accelerator, vae_dtype):
        sample_parameters = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)
        vae = None
        if args.sample_prompts or args.depth_anchor_weight > 0:
            vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False).eval()
        return sample_parameters, vae

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        del dit_dtype, generator, image_path, control_video_path
        if frame_count not in {1, 5}:
            raise ValueError("Experimental MiniMax-H3 training previews support only 1 or 5 frames")
        if do_classifier_free_guidance or guidance_scale != 1.0 or cfg_scale not in {None, 1.0}:
            raise ValueError("MiniMax-H3 is guidance-distilled; preview CFG must remain 1.0")
        seed = int(sample_parameter.get("seed", torch.initial_seed()))
        shift = float(sample_parameter.get("discrete_flow_shift", discrete_flow_shift or 12.0))
        device = accelerator.device
        if frame_count == 1:
            latent = sample_image_latent(
                transformer,
                sample_parameter["mmh3_hidden_states"],
                width=width,
                height=height,
                steps=sample_steps,
                seed=seed,
                shift=shift,
                device=device,
                dtype=torch.bfloat16,
                picture_model=self.architecture == ARCHITECTURE_MINIMAX_H3_PICTURE,
            )
        else:
            logger.info("Generating experimental five-frame MiniMax-H3 scheduled preview")
            latent = sample_video_latent(
                transformer,
                sample_parameter["mmh3_hidden_states"],
                frame_count=frame_count,
                width=width,
                height=height,
                steps=sample_steps,
                seed=seed,
                device=device,
                video_shift=shift,
                audio_shift=3.0,
            )
        latent = latent.detach().to("cpu")
        clean_memory_on_device(device)

        minimum_free = int(float(args.minimax_h3_preview_decode_min_free_gb) * 1024**3)
        parked = transformer.park_resident_block_weights_for_decode(minimum_free)
        if parked:
            logger.info("MiniMax-H3 preview temporarily parked %s resident base weights for VAE decode", len(parked))
        try:
            decode_device = self._get_depth_vae_device(args, device) if args.depth_anchor_weight > 0 else device
            vae.to(device=decode_device, dtype=torch.float16).eval()
            with torch.no_grad():
                decode_latent = latent.to(device=decode_device, dtype=torch.float16)
                if frame_count == 1:
                    pixels = decode_image_latent(vae, decode_latent, single_frame=self.architecture == ARCHITECTURE_MINIMAX_H3_PICTURE).cpu()
                else:
                    video = decode_video_latent(vae, decode_latent, frame_count=frame_count).cpu()
                    pixels = video.permute(3, 0, 1, 2).unsqueeze(0).float().div_(255.0)
        finally:
            if not (args.depth_anchor_weight > 0 and args.keep_depth_vae_on_device):
                vae.to("cpu")
            if args.depth_anchor_weight > 0 and self._depth_vae_device != device:
                clean_memory_on_device(self._depth_vae_device)
            clean_memory_on_device(device)
            transformer.restore_parked_block_weights(parked)
            clean_memory_on_device(device)
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        del vae_dtype
        logger.info("Loading MiniMax-H3 preview VAE on CPU from %s", vae_path)
        return load_video_vae(
            vae_path,
            device="cpu",
            dtype=torch.float16,
            disable_mmap=args.disable_numpy_memmap,
            picture_model=self.architecture == ARCHITECTURE_MINIMAX_H3_PICTURE,
        )

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        model = load_h3_transformer(
            dit_path,
            device=loading_device,
            dtype=torch.bfloat16,
            attn_mode=attn_mode,
            split_attn=split_attn,
            disable_mmap=args.disable_numpy_memmap,
            convrot_bwd_mode=args.convrot_bwd_mode,
            picture_model=self.architecture == ARCHITECTURE_MINIMAX_H3_PICTURE,
        )
        if not model._has_convrot_int8:
            raise ValueError(
                "This experimental trainer accepts only the pruned ConvRot INT8 MiniMax-H3 checkpoint; "
                "it intentionally does not require or convert the ~66 GB BF16 transformer"
            )
        return model

    def compile_transformer(self, args, transformer):
        return transformer

    def scale_shift_latents(self, latents):
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        if latents.ndim != 5 or tuple(latents.shape[1:3]) != (24, 1):
            raise ValueError(f"MiniMax-H3 image latents must be [B,24,1,H,W], got {tuple(latents.shape)}")
        if latents.shape[-2] % 2 or latents.shape[-1] % 2:
            raise ValueError("MiniMax-H3 cached latent height and width must be divisible by 2 (image buckets by 32 pixels)")
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        if latents.shape[0] != 1:
            raise ValueError("MiniMax-H3 image training requires batch_size=1")
        hidden_rows = batch.get("mmh3_hidden_states")
        if not isinstance(hidden_rows, list) or len(hidden_rows) != 1:
            raise ValueError("MiniMax-H3 text cache must provide one varlen_mmh3_hidden_states tensor")
        text_hidden_states = hidden_rows[0].unsqueeze(0).to(
            device=accelerator.device,
            dtype=network_dtype,
        )
        model_t = 1.0 - (timesteps.to(accelerator.device, dtype=torch.float32) - 1.0) / 1000.0
        with accelerator.autocast():
            prediction = transformer.forward_image(
                noisy_model_input.to(accelerator.device, dtype=network_dtype),
                model_t,
                text_hidden_states,
            )
        target = latents.to(accelerator.device, dtype=prediction.dtype) - noise.to(
            accelerator.device,
            dtype=prediction.dtype,
        )
        return DiTOutput(pred=prediction, target=target)

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        noisy_input, timesteps = self.get_noisy_model_input_and_timesteps(
            args,
            noise,
            latents,
            batch["timesteps"],
            noise_scheduler,
            accelerator.device,
            dit_dtype,
        )
        base_sigma = base_sigma_from_model_timestep(args, timesteps, latents)
        sigma_gate_open = bool(
            torch.any(base_sigma >= getattr(args, "h3_guidance_distillation_sigma_min", 0.15)).item()
        )
        unconditional_prediction = None
        run_dynamic_sigma = guidance_protection_enabled(args) and should_preserve_base(
            global_step, getattr(args, "h3_dynamic_sigma_every_n_steps", 1)
        ) and sigma_gate_open
        if run_dynamic_sigma:
            unconditional_rows = batch.get("mmh3_unconditional_hidden_states")
            if not isinstance(unconditional_rows, list) or len(unconditional_rows) != 1:
                raise ValueError(
                    "MiniMax-H3 guidance protection requires cached empty-prompt text states. "
                    "Enable Rebuild Caption/Text Cache and run again."
                )
            unconditional_batch = dict(batch)
            unconditional_batch["mmh3_hidden_states"] = unconditional_rows
            with torch.no_grad():
                unconditional_prediction = self.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents,
                    unconditional_batch,
                    noise,
                    noisy_input,
                    timesteps,
                    network_dtype,
                ).pred.detach()
        reference_prediction = None
        run_base_preservation = (
            getattr(args, "h3_base_preservation_enabled", False)
            and should_preserve_base(global_step, getattr(args, "h3_base_preservation_every_n_steps", 10))
        )
        if run_base_preservation:
            unwrapped_network = accelerator.unwrap_model(network)
            original_multiplier = float(getattr(unwrapped_network, "multiplier", 1.0))
            unwrapped_network.set_multiplier(0.0)
            disable_assistant = (
                self._training_assistant is not None
                and getattr(args, "h3_base_preservation_reference", "assistant") == "base"
            )
            if disable_assistant:
                self._training_assistant.set_enabled(False)
            try:
                with torch.no_grad():
                    reference_prediction = self.call_dit(
                        args,
                        accelerator,
                        transformer,
                        latents,
                        batch,
                        noise,
                        noisy_input,
                        timesteps,
                        network_dtype,
                    ).pred.detach()
            finally:
                if disable_assistant:
                    self._training_assistant.set_enabled(True)
                unwrapped_network.set_multiplier(original_multiplier)
        output = self.call_dit(
            args,
            accelerator,
            transformer,
            latents,
            batch,
            noise,
            noisy_input,
            timesteps,
            network_dtype,
        )
        teacher_conditioned = False
        ordinary_target = output.target
        if getattr(args, "h3_teacher_matching", False):
            teacher_rows = batch.get("mmh3_teacher_ref_hidden_states")
            teacher_tags = batch.get("mmh3_teacher_ref_token_tags")
            if not isinstance(teacher_rows, list) or len(teacher_rows) != 1:
                raise ValueError(
                    "Compact reference-guided learning needs teacher text rows. Rebuild Caption/Text Cache with it enabled."
                )
            if not isinstance(teacher_tags, list) or len(teacher_tags) != 1:
                raise ValueError("Compact reference-guided cache is missing visual token tags")
            unwrapped_network = accelerator.unwrap_model(network)
            base_sigma_value = float(base_sigma.reshape(-1)[0].item())
            teacher_conditioned = base_sigma_value <= getattr(args, "h3_teacher_condition_sigma_max", 0.75)
            unwrapped_network.set_enabled(False)
            try:
                with torch.no_grad(), accelerator.autocast():
                    if teacher_conditioned:
                        # The released packed H3 path has a two-latent-frame minimum. Repeat the
                        # noisy still only for this frozen teacher and keep its first prediction;
                        # the ordinary one-frame student path and its gradients remain unchanged.
                        teacher_video = noisy_input.to(accelerator.device, dtype=network_dtype).repeat(1, 1, 2, 1, 1)
                        if getattr(args, "h3_teacher_conditions", "ref") == "subject_ref":
                            ref_keys = sorted(
                                key for key in batch if key.startswith("latents_ref_") and key.endswith("_image")
                            )
                            if not ref_keys:
                                raise ValueError(
                                    "Other-subject teacher needs cached control_path/control_path_N reference images"
                                )
                            reference_videos = tuple(
                                batch[key].to(accelerator.device, dtype=network_dtype) for key in ref_keys
                            )
                        else:
                            reference_videos = (latents.to(accelerator.device, dtype=network_dtype),)
                        text = teacher_rows[0].unsqueeze(0).to(accelerator.device, dtype=network_dtype)
                        tags = teacher_tags[0].unsqueeze(0).to(accelerator.device, dtype=torch.int64)
                        references = tuple(
                            H3ReferenceGeometry("image", video=H3VideoGeometry(*reference.shape[2:]))
                            for reference in reference_videos
                        )
                        layout = build_h3_layout(
                            task="ref2va",
                            text_length=text.shape[1],
                            target_video=H3VideoGeometry(*teacher_video.shape[2:]),
                            target_audio_frames=8,
                            references=references,
                        )
                        silent_audio = torch.zeros(
                            (1, 32, 2, 8), device=accelerator.device, dtype=network_dtype
                        )
                        teacher_model_t = 1.0 - (
                            timesteps.to(accelerator.device, dtype=torch.float32) - 1.0
                        ) / 1000.0
                        teacher_prediction = transformer(
                            video_latents=teacher_video,
                            audio_latents=silent_audio,
                            text_hidden_states=text,
                            text_token_tags=tags,
                            layout=layout,
                            model_t_video=teacher_model_t,
                            model_t_audio=teacher_model_t,
                            visual_condition_latents=reference_videos,
                        ).video[:, :, :1].detach()
                    else:
                        teacher_prediction = transformer.forward_image(
                            noisy_input.to(accelerator.device, dtype=network_dtype),
                            1.0 - (timesteps.to(accelerator.device, dtype=torch.float32) - 1.0) / 1000.0,
                            batch["mmh3_hidden_states"][0].unsqueeze(0).to(accelerator.device, dtype=network_dtype),
                        ).detach()
            finally:
                unwrapped_network.set_enabled(True)
            output.target = teacher_prediction.to(output.pred.dtype)
        if unconditional_prediction is not None:
            normal_target = output.target
            output.target = build_guided_target(
                unconditional_prediction,
                normal_target,
                args.h3_guidance_distillation_scale,
                sigma=timesteps.float() / 1000.0,
                schedule=args.h3_guidance_distillation_schedule,
            )
        if "image_loss_mask" in batch:
            output.extra["image_loss_mask"] = batch["image_loss_mask"]
        diffusion_loss, metrics = self.compute_loss(
            args,
            output,
            timesteps,
            noise_scheduler,
            dit_dtype,
            network_dtype,
            global_step,
        )
        total = diffusion_loss
        if getattr(args, "h3_teacher_matching", False):
            metrics["teacher/conditioned"] = torch.tensor(
                1.0 if teacher_conditioned else 0.0, device=output.pred.device
            )
            metrics["teacher/flow_gap_rms"] = (
                output.target.float() - ordinary_target.float()
            ).pow(2).mean().sqrt().detach()
            direct_strength = float(getattr(args, "h3_teacher_direct_loss_weight", 0.0))
            if teacher_conditioned and direct_strength > 0.0:
                from musubi_tuner.dataset.image_loss_masks import image_mse_loss

                direct = image_mse_loss(output.pred.float(), ordinary_target.float(), batch.get("image_loss_mask"),
                                        getattr(args, "image_mask_background_weight", 0.1))
                scale = diffusion_loss.detach() / direct.detach().clamp_min(1e-12)
                direct_scaled = direct * scale
                total = total + direct_strength * direct_scaled
                metrics["loss/teacher"] = diffusion_loss.detach()
                metrics["loss/direct"] = direct.detach()
                metrics["loss/direct_scaled"] = direct_scaled.detach()
        if reference_prediction is not None:
            preservation = base_preservation_loss(output.pred, reference_prediction)
            total = total + getattr(args, "h3_base_preservation_loss_weight", 0.0) * preservation
            metrics["loss/h3_base_preservation"] = preservation.detach()
        if guidance_protection_enabled(args):
            metrics["guidance/base_sigma"] = base_sigma.mean().detach()
            metrics["guidance/applied"] = torch.tensor(
                1.0 if unconditional_prediction is not None else 0.0, device=output.pred.device
            )
        if unconditional_prediction is not None:
            metrics.update(
                {
                    "loss/h3_guidance_target_delta": torch.nn.functional.mse_loss(
                        output.target.float(), normal_target.float()
                    ).detach(),
                }
            )
        run_depth = args.depth_anchor_weight > 0 and global_step % args.depth_anchor_every_n_steps == 0
        if run_depth:
            from musubi_tuner.perceptual.depth_anchor import (
                DepthAnchor,
                reconstruct_clean_latents_h3,
                resize_latents_for_depth_decode,
            )

            predicted_clean = reconstruct_clean_latents_h3(noisy_input, output.pred, timesteps)
            predicted_for_depth = resize_latents_for_depth_decode(
                predicted_clean,
                args.depth_anchor_input_size,
                spatial_compression_ratio=16,
            )
            vae_device = self._get_depth_vae_device(args, accelerator.device)
            vae.to(vae_device, dtype=torch.float16).requires_grad_(False)
            predicted_pixels = decode_image_latent(
                vae,
                predicted_for_depth.to(vae_device, dtype=torch.float16),
                checkpoint_decode=args.depth_anchor_grad_checkpoint,
            )[:, :, 0].to(accelerator.device)

            if self._depth_anchor is None:
                logger.info("Loading frozen depth perceptor: %s", args.depth_anchor_model)
                self._depth_anchor = DepthAnchor(
                    args.depth_anchor_model,
                    accelerator.device,
                    input_size=args.depth_anchor_input_size,
                    grad_checkpoint=args.depth_anchor_grad_checkpoint,
                )
            else:
                self._depth_anchor.to(accelerator.device)

            target_depths = []
            for sample in latents:
                key = self._depth_anchor.cache_key(sample)
                if key in self._depth_anchor.target_cache:
                    target_depths.append(self._depth_anchor.target_depth(None, key))
                    continue
                with torch.no_grad():
                    target_latent = resize_latents_for_depth_decode(
                        sample.unsqueeze(0),
                        args.depth_anchor_input_size,
                        spatial_compression_ratio=16,
                    )
                    target_pixels = decode_image_latent(
                        vae,
                        target_latent.to(vae_device, dtype=torch.float16),
                    )[:, :, 0]
                    target_pixels = target_pixels.to(accelerator.device)
                target_depths.append(self._depth_anchor.target_depth(target_pixels, key))
            target_depth = torch.cat(target_depths, dim=0)
            depth_loss = self._depth_anchor.loss(
                predicted_pixels,
                grad_weight=args.depth_anchor_gradient_weight,
                target_depth=target_depth,
            )
            total = total + float(args.depth_anchor_weight) * depth_loss
            metrics.update(
                {
                    "loss/diffusion": diffusion_loss.detach(),
                    "loss/depth_anchor": depth_loss.detach(),
                    "loss/depth_anchor_weighted": (float(args.depth_anchor_weight) * depth_loss).detach(),
                }
            )

        self._dop_step_context = (
            batch,
            latents,
            noise,
            noisy_input,
            timesteps,
            network_dtype,
        ) if dop_enabled(args) else None
        return total, metrics

    def on_after_primary_backward(self, args, accelerator, vae):
        if args.depth_anchor_weight <= 0:
            return
        if not args.keep_depth_vae_on_device:
            vae.to("cpu")
            if self._depth_vae_device is not None:
                clean_memory_on_device(self._depth_vae_device)
        if self._depth_anchor is not None and not args.keep_depth_helpers_on_gpu:
            self._depth_anchor.to("cpu")
        clean_memory_on_device(accelerator.device)

    def compute_auxiliary_loss(self, args, accelerator, transformer, network):
        if self._dop_step_context is None:
            return None, {}
        batch, latents, noise, noisy_input, timesteps, network_dtype = self._dop_step_context
        self._dop_step_context = None
        return compute_dop_loss(
            self,
            args,
            accelerator,
            transformer,
            network,
            batch,
            latents,
            noise,
            noisy_input,
            timesteps,
            network_dtype,
            embedding_key="mmh3_hidden_states",
            dop_embedding_key="dop_mmh3_hidden_states",
        )

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
        del transformer, global_step
        if not sync_gradients or args.weight_noise_sigma <= 0:
            return
        if self._weight_noise is None:
            from musubi_tuner.training.weight_noise import AdapterWeightNoise, WeightNoiseConfig

            self._weight_noise = AdapterWeightNoise(
                WeightNoiseConfig(args.weight_noise_sigma, args.weight_noise_mode, args.weight_noise_bound_norm)
            )
        self._weight_noise_logs = self._weight_noise.apply(accelerator.unwrap_model(network).get_trainable_params())

    def extra_step_logs(self, args, logs):
        del args, logs
        return dict(self._weight_noise_logs)

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        return {
            "ss_minimax_h3_training_mode": "experimental_image_only",
            "ss_minimax_h3_base_format": "pruned_int8_convrot",
            "ss_minimax_h3_convrot_bwd_mode": args.convrot_bwd_mode,
            "ss_minimax_h3_weight_noise_sigma": args.weight_noise_sigma,
            "ss_minimax_h3_weight_noise_mode": args.weight_noise_mode,
            "ss_minimax_h3_weight_noise_bound_norm": args.weight_noise_bound_norm,
            "ss_minimax_h3_depth_anchor_weight": args.depth_anchor_weight,
            "ss_minimax_h3_depth_anchor_model": args.depth_anchor_model if args.depth_anchor_weight > 0 else "",
            "ss_minimax_h3_depth_anchor_input_size": args.depth_anchor_input_size,
            "ss_minimax_h3_depth_anchor_gradient_weight": args.depth_anchor_gradient_weight,
            "ss_minimax_h3_keep_depth_helpers_on_gpu": args.keep_depth_helpers_on_gpu,
            "ss_dop_loss_weight": args.dop_loss_weight,
            "ss_dop_trigger_word": args.dop_trigger_word if dop_enabled(args) else "",
            "ss_dop_class_word": args.dop_class_word if dop_enabled(args) else "",
            "ss_minimax_h3_guidance_distillation_protection": guidance_protection_enabled(args),
            "ss_minimax_h3_guidance_distillation_scale": args.h3_guidance_distillation_scale,
            "ss_minimax_h3_guidance_distillation_schedule": args.h3_guidance_distillation_schedule,
            "ss_minimax_h3_guidance_distillation_sigma_min": getattr(
                args, "h3_guidance_distillation_sigma_min", 0.15
            ),
            "ss_minimax_h3_quality_protection_method": getattr(args, "h3_quality_protection_method", None) or "legacy",
            "ss_minimax_h3_training_assistant": str(getattr(args, "h3_training_assistant", "") or ""),
            "ss_minimax_h3_base_preservation_loss_weight": getattr(args, "h3_base_preservation_loss_weight", 0.0),
            "ss_minimax_h3_base_preservation_every_n_steps": getattr(args, "h3_base_preservation_every_n_steps", 10),
            "ss_minimax_h3_training_assistant_enabled": getattr(args, "h3_training_assistant_enabled", False),
            "ss_minimax_h3_dynamic_sigma_every_n_steps": getattr(args, "h3_dynamic_sigma_every_n_steps", 1),
            "ss_minimax_h3_base_preservation_enabled": getattr(args, "h3_base_preservation_enabled", False),
            "ss_minimax_h3_base_preservation_reference": getattr(args, "h3_base_preservation_reference", "assistant"),
            "ss_minimax_h3_foundation_lora": str(getattr(args, "h3_foundation_lora", "") or ""),
            "ss_minimax_h3_foundation_lora_multiplier": getattr(args, "h3_foundation_lora_multiplier", 1.0),
        }


class MiniMaxH3PictureNetworkTrainer(MiniMaxH3ImageNetworkTrainer):
    """LoRA training for the dedicated six-head MiniMax H3 Image checkpoint."""

    process_batch = NetworkTrainer.process_batch

    @property
    def architecture(self):
        return ARCHITECTURE_MINIMAX_H3_PICTURE

    @property
    def architecture_full_name(self):
        return ARCHITECTURE_MINIMAX_H3_PICTURE_FULL

    def handle_model_specific_args(self, args):
        super().handle_model_specific_args(args)
        if (guidance_protection_enabled(args) or args.h3_base_preservation_enabled or args.depth_anchor_weight
                or dop_enabled(args) or args.h3_teacher_matching or args.h3_foundation_lora or args.base_weights):
            raise ValueError("MiniMax H3 Image currently supports direct text-to-image LoRA training and an optional frozen Image training adapter.")
        args.timestep_sampling = "uniform"
        args.num_timestep_buckets = None
        args.show_timesteps = None
        args.weighting_scheme = "none"

    def _build_dataset(self, args):
        from musubi_tuner.dataset.image_video_dataset import ImageDataset

        group, collator, epoch = super()._build_dataset(args)
        if any(not isinstance(dataset, ImageDataset) or dataset.has_control for dataset in group.datasets):
            raise ValueError("MiniMax H3 Image needs captioned images without control images.")
        return group, collator, epoch

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        grid = torch.tensor(H3_PICTURE_SIGMAS[:-1], device=device)
        index = torch.randint(len(grid), (latents.shape[0],), device=device)
        sigma = grid[index]
        shaped = sigma.reshape(-1, *(1 for _ in latents.shape[1:]))
        return torch.lerp(latents.float(), noise.float(), shaped).to(dtype), sigma * 1000.0 + 1.0

    def extra_metadata(self, args):
        metadata = super().extra_metadata(args)
        metadata["ss_minimax_h3_training_mode"] = "minimax_h3_image_t2i"
        metadata["ss_minimax_h3_pdd_heads"] = len(H3_PICTURE_SIGMAS) - 1
        return metadata


def minimax_h3_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--h3_picture_model", action="store_true", help="train the dedicated MiniMax H3 Image checkpoint")
    foundation = parser.add_argument_group("MiniMax-H3 frozen foundation LoRA")
    foundation.add_argument(
        "--h3_foundation_lora",
        type=str,
        default=None,
        help="ordinary MiniMax-H3 LoRA kept frozen and active while a separate new LoRA trains on top",
    )
    foundation.add_argument("--h3_foundation_lora_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--convrot_bwd_mode",
        choices=("bf16", "int8"),
        default="bf16",
        help="ConvRot base-weight backward: bf16 is the safe default; int8 requires Triton and is more experimental",
    )
    weight_noise = parser.add_argument_group("MiniMax-H3 adapter regularization")
    weight_noise.add_argument("--weight_noise_sigma", type=float, default=0.0)
    weight_noise.add_argument("--weight_noise_mode", choices=("relative", "absolute"), default="relative")
    weight_noise.add_argument("--weight_noise_bound_norm", action="store_true")
    guidance = parser.add_argument_group("MiniMax-H3 guidance-distillation protection")
    guidance.add_argument("--h3_guidance_distillation_protection", action="store_true")
    guidance.add_argument("--h3_dynamic_sigma_every_n_steps", type=int, default=1)
    guidance.add_argument("--h3_guidance_distillation_scale", type=float, default=4.0)
    guidance.add_argument("--h3_guidance_distillation_schedule", choices=("sigma", "constant"), default="sigma")
    guidance.add_argument(
        "--h3_guidance_distillation_sigma_min",
        type=float,
        default=0.15,
        help="skip the extra Dynamic Sigma forward below this pre-shift base sigma (recommended: 0.15; 0 = always)",
    )
    guidance.add_argument(
        "--h3_quality_protection_method",
        choices=("dynamic", "assistant", "assistant_preservation", "off"),
        default=None,
    )
    guidance.add_argument("--h3_training_assistant", default=DEFAULT_ASSISTANT)
    guidance.add_argument("--h3_training_assistant_enabled", action="store_true")
    guidance.add_argument("--h3_base_preservation_enabled", action="store_true")
    guidance.add_argument("--h3_base_preservation_loss_weight", type=float, default=0.0)
    guidance.add_argument("--h3_base_preservation_every_n_steps", type=int, default=10)
    guidance.add_argument("--h3_base_preservation_reference", choices=("base", "assistant"), default="assistant")
    teacher = parser.add_argument_group("MiniMax-H3 compact reference-guided learning (experimental)")
    teacher.add_argument("--h3_teacher_matching", action="store_true")
    teacher.add_argument("--h3_teacher_conditions", choices=("ref",), default="ref")
    teacher.add_argument("--h3_teacher_condition_sigma_max", type=float, default=0.75)
    teacher.add_argument("--h3_teacher_direct_loss_weight", type=float, default=0.0)
    depth_anchor = parser.add_argument_group("MiniMax-H3 perceptual depth anchor (experimental)")
    depth_anchor.add_argument("--depth_anchor_weight", type=float, default=0.0)
    depth_anchor.add_argument("--depth_anchor_model", default="depth-anything/Depth-Anything-V2-Small-hf")
    depth_anchor.add_argument("--depth_anchor_input_size", type=int, default=518)
    depth_anchor.add_argument("--depth_anchor_gradient_weight", type=float, default=0.5)
    depth_anchor.add_argument("--depth_anchor_grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    depth_anchor.add_argument("--keep_depth_helpers_on_gpu", action="store_true")
    depth_anchor.add_argument(
        "--depth_anchor_vae_device",
        default="training",
        help="Device for differentiable MiniMax VAE decoding: training, secondary, or a logical CUDA device",
    )
    depth_anchor.add_argument("--keep_depth_vae_on_device", action="store_true")
    depth_anchor.add_argument("--depth_anchor_every_n_steps", type=int, default=1)
    parser.add_argument(
        "--text_encoder",
        default=None,
        help="compact Qwen3-VL safetensors used only to pre-encode training preview prompts",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_PROCESSOR_ID, help="Qwen3-VL tokenizer repo or local directory")
    parser.add_argument(
        "--text_encoder_load_mode",
        choices=("auto", "direct", "nf4"),
        default="auto",
        help="compact MiniMax-H3 text encoder loading mode for previews",
    )
    parser.add_argument(
        "--text_encoder_blocks_to_swap",
        type=int,
        default=0,
        help="Stream 0-50 preview text-encoder layers from system RAM; 50 uses the least VRAM",
    )
    parser.add_argument(
        "--minimax_h3_preview_decode_min_free_gb",
        type=float,
        default=9.0,
        help="free-VRAM target before moving the MiniMax-H3 VAE to CUDA for a preview",
    )
    parser.set_defaults(
        network_module="musubi_tuner.networks.lora_minimax_h3",
        timestep_sampling="krea2_shift",
        blocks_to_swap=30,
        block_swap_h2d_only=True,
        gradient_checkpointing=True,
        sdpa=True,
    )
    return parser


def main() -> None:
    parser = minimax_h3_image_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"
    trainer = MiniMaxH3PictureNetworkTrainer() if args.h3_picture_model else MiniMaxH3ImageNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
