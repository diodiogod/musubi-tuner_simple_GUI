"""MOSS LoRA training through Musubi's optimizer, dataset and resume runtime."""
import gc
import math
import torch

from musubi_tuner.hv_train_network import NetworkTrainer, DiTOutput, setup_parser_common, read_config_from_file, load_prompts
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE, ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.moss_image.vae import load_vae
from musubi_tuner.moss_image.checkpoint import load_model
from musubi_tuner.moss_image.text_encoder import load_text_encoder, load_tokenizer, encode_prompts, pad_text_embeds
from musubi_tuner.moss_image.sampling import generate_latents, parse_skip_layers
from musubi_tuner.training.dop import compute_dop_loss, dop_enabled, validate_dop_config
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils import model_utils


class MossImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1
        self._dop_step_context = None
        self._depth_regularizer = None
        self._weight_noise = None
        self._weight_noise_logs = {}
        self._repa = None
        self._repa_capture = None
        self._repa_dataset = None
        self._repa_targets = None

    @property
    def architecture(self):
        return ARCHITECTURE_MOSS_IMAGE

    @property
    def architecture_full_name(self):
        return ARCHITECTURE_MOSS_IMAGE_FULL

    def handle_model_specific_args(self, args):
        from musubi_tuner.moss_image.repa import validate_config
        validate_config(args)
        self.dit_dtype = torch.bfloat16
        self._i2v_training = self._control_training = False
        self.default_guidance_scale = 3.0
        if args.depth_anchor_weight < 0 or args.weight_noise_sigma < 0:
            raise ValueError("Depth strength and adapter noise must be non-negative.")
        if args.depth_anchor_input_size <= 0 or args.depth_anchor_input_size % 14:
            raise ValueError("Depth input size must be a positive multiple of 14.")
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("MOSS currently supports BF16 and ConvRot INT8 weights, not FP8.")
        if args.moss_pretraining_schedule:
            if not math.isfinite(args.moss_logit_mean) or not math.isfinite(args.moss_logit_std) or args.moss_logit_std <= 0:
                raise ValueError("MOSS logit mean must be finite and logit standard deviation must be positive.")
            if not 0 < args.moss_timestep_cutoff <= 1 or not 0 <= args.moss_timestep_uniform_ratio <= 1:
                raise ValueError("MOSS timestep cutoff must be in (0, 1] and uniform ratio in [0, 1].")
            if args.min_timestep is not None or args.max_timestep is not None:
                raise ValueError("MOSS pretraining schedule does not support a restricted timestep range; disable it to use general timestep controls.")
        if dop_enabled(args):
            validate_dop_config(args.dop_trigger_word, args.dop_class_word, args.dop_loss_weight)

    def load_vae(self, args, vae_dtype, vae_path):
        return load_vae(vae_path, dtype=torch.float32, device="cpu", disable_mmap=args.disable_mmap).eval()

    def _build_dataset(self, args):
        from musubi_tuner.dataset.image_video_dataset import ImageDataset
        dataset, collator, epoch = super()._build_dataset(args)
        if any(not isinstance(item, ImageDataset) for item in dataset.datasets):
            raise ValueError("MOSS-Image supports image datasets only. Remove video datasets from this run.")
        if getattr(args, "moss_repa_enabled", False):
            self._repa_dataset = dataset
        return dataset, collator, epoch

    def _prepare_sampling(self, args, accelerator, vae_dtype):
        # Depth decoding also needs the VAE when no sample prompts were provided.
        prompts = self.process_sample_prompts(args, accelerator, args.sample_prompts) if args.sample_prompts else None
        vae = None
        if args.sample_prompts or args.depth_anchor_weight > 0 or getattr(args, "moss_repa_enabled", False):
            if not args.vae:
                raise ValueError("A Flux.2 VAE is required for MOSS previews, depth anchoring and REPA.")
            vae = self.load_vae(args, vae_dtype, args.vae).requires_grad_(False).eval()
        if getattr(args, "moss_repa_enabled", False):
            from musubi_tuner.moss_image.repa_cache import MossRepaTargets
            if accelerator.num_processes != 1:
                raise ValueError("Experimental MOSS REPA currently supports a single training GPU/process.")
            self._repa_targets = MossRepaTargets(args, accelerator.device)
            self._repa_targets.prepare(self._repa_dataset, vae)
            self._repa_dataset = None
            if not args.sample_prompts and args.depth_anchor_weight <= 0:
                vae = None  # Prepared REPA targets need no live decoder.
                gc.collect()
        return prompts, vae

    def load_transformer(self, accelerator, args, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype):
        return load_model(dit_path, device=loading_device, dtype=torch.bfloat16,
                          attn_mode=attn_mode, split_attn=split_attn, convrot_int8=args.convrot_int8,
                          convrot_bwd_mode=args.convrot_bwd_mode, disable_mmap=args.disable_mmap)

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(args, transformer, [transformer.layers], disable_linear=self.blocks_to_swap > 0)

    def scale_shift_latents(self, latents):
        return latents

    def extra_trainable_params(self, args, accelerator, network, transformer, trainable_params):
        if not getattr(args, "moss_repa_enabled", False):
            return trainable_params
        from musubi_tuner.moss_image.repa import MossRepa, validate_config
        if accelerator.num_processes != 1:
            raise ValueError("Experimental MOSS REPA currently supports a single training GPU/process.")
        model = accelerator.unwrap_model(transformer)
        validate_config(args, model.num_layers)
        self._repa = MossRepa(args, model.dim, accelerator.device, feature_cache=self._repa_targets)
        # Registered before load_state, so true resumes restore the learned head.
        # Kept outside the adapter: generation LoRA files contain no REPA weights.
        accelerator.register_for_checkpointing(self._repa)
        # Reuse the existing learning-rate group so the shared tracker keeps
        # matching parameter groups to its lr_descriptions (including LoKr).
        groups = [dict(group) for group in trainable_params]
        groups[0]["params"] = list(groups[0]["params"]) + list(self._repa.projector.parameters())
        return groups

    def extra_metadata(self, args):
        if not getattr(args, "moss_repa_enabled", False):
            return {}
        return {"ss_" + key: str(getattr(args, key)) for key in
                ("moss_repa_enabled", "moss_repa_weight", "moss_repa_encoder", "moss_repa_block",
                 "moss_repa_image_size", "moss_repa_device", "moss_repa_cache_size", "moss_repa_cache_dir")}

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        if not args.moss_pretraining_schedule:
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)
        # OpenMOSS technical report Eq. 4, as implemented in Kohya sd-scripts
        # PR #2471: inverse-CDF truncated logit normal, rescaled and snapped to
        # the original 1000-step grid, with a small uniform mixture.
        count = noise.shape[0]
        mean = torch.full((count,), args.moss_logit_mean, dtype=torch.float64, device=device)
        cutoff = args.moss_timestep_cutoff
        if cutoff < 1:
            lower = -math.log(cutoff / (1 - cutoff))
            p_min = torch.special.ndtr((lower - mean) / args.moss_logit_std)
        else:
            p_min = torch.zeros_like(mean)
        u = torch.rand(count, dtype=torch.float64, device=device) if timesteps is None else torch.as_tensor(timesteps, dtype=torch.float64, device=device)
        p = (p_min + u * (1 - p_min)).clamp(1e-12, 1 - 1e-12)
        z = mean + args.moss_logit_std * torch.special.ndtri(p)
        sigma = (torch.ceil(torch.sigmoid(-z) / cutoff * 1000).clamp(1, 1000) / 1000).float()
        if args.moss_timestep_uniform_ratio:
            random_steps = torch.randint(1, 1001, (count,), device=device).float() / 1000
            sigma = torch.where(torch.rand(count, device=device) < args.moss_timestep_uniform_ratio, random_steps, sigma)
        noisy = (1 - sigma.view(-1, 1, 1, 1, 1)) * latents + sigma.view(-1, 1, 1, 1, 1) * noise
        return noisy.to(dtype), sigma * 1000

    @staticmethod
    def _model_timesteps(args, timesteps):
        # Musubi's continuous samplers add one to the time after mixing noise.
        # MOSS conditions on sigma*1000. Its grid sampler and the shared sigma
        # sampler already return that exact value.
        if args.moss_pretraining_schedule or args.timestep_sampling == "sigma":
            return timesteps
        return timesteps - 1

    def compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
        sigma = (self._model_timesteps(args, timesteps).float() / 1000).clamp_min(1e-6).view(-1, 1, 1, 1, 1)
        loss = torch.nn.functional.mse_loss(output.pred.float(), output.target.float(), reduction="none")
        if args.weighting_scheme == "sigma_sqrt":
            loss = loss * sigma.pow(-2)
        elif args.weighting_scheme == "cosmap":
            loss = loss * (2 / (math.pi * (1 - 2 * sigma + 2 * sigma.square())))
        return loss.mean(), {}

    def call_dit(self, args, accelerator, transformer, latents, batch, noise,
                 noisy_model_input, timesteps, network_dtype, **kwargs):
        if latents.ndim != 5 or latents.shape[1:3] != (128, 1):
            raise ValueError("MOSS training requires packed image latents [B,128,1,H,W]. Recache with the MOSS VAE tool.")
        context, mask = pad_text_embeds(batch["moss_embed"])
        context = context.to(accelerator.device, dtype=network_dtype)
        mask = mask.to(accelerator.device)
        inputs = noisy_model_input.squeeze(2).to(accelerator.device, dtype=network_dtype)
        if args.gradient_checkpointing:
            inputs.requires_grad_(True)
            context.requires_grad_(True)
        with accelerator.autocast():
            capture = ({"feature_layer": args.moss_repa_block, "feature_buffer": self._repa_capture}
                       if self._repa_capture is not None else {})
            prediction = transformer(inputs, self._model_timesteps(args, timesteps), context, mask, **capture).unsqueeze(2)
        return DiTOutput(pred=prediction, target=noise - latents)

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise,
                      noise_scheduler, dit_dtype, network_dtype, vae, global_step):
        noisy, times = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype,
        )
        target = self._repa.targets(vae, latents) if self._repa is not None else None
        features = {} if self._repa is not None else None
        self._repa_capture = features
        try:
            output = self.call_dit(args, accelerator, transformer, latents, batch, noise, noisy, times, network_dtype)
        finally:
            self._repa_capture = None  # DOP and previews must never capture features.
        loss, metrics = self.compute_loss(args, output, times, noise_scheduler, dit_dtype, network_dtype, global_step)
        if self._repa is not None:
            repa_loss, repa_metrics = self._repa.loss(features.pop("image"), target, *latents.shape[-2:])
            metrics.update({"loss/diffusion": loss.detach(), **repa_metrics})
            loss = loss + args.moss_repa_weight * repa_loss
        if args.depth_anchor_weight > 0:
            if self._depth_regularizer is None:
                from musubi_tuner.moss_image.regularization import MossDepthRegularizer
                self._depth_regularizer = MossDepthRegularizer(args, accelerator.device)
            depth_loss = self._depth_regularizer.loss(vae, latents, noisy, output.pred, self._model_timesteps(args, times))
            metrics.setdefault("loss/diffusion", loss.detach())
            metrics["loss/depth_anchor"] = depth_loss.detach()
            loss = loss + args.depth_anchor_weight * depth_loss
        self._dop_step_context = (batch, latents, noise, noisy, times, network_dtype) if dop_enabled(args) else None
        return loss, metrics

    def on_after_primary_backward(self, args, accelerator, vae):
        if self._repa is not None and accelerator.sync_gradients and args.max_grad_norm:
            # The shared trainer clips adapter parameters separately. The head
            # is not part of the exported adapter's get_trainable_params().
            torch.nn.utils.clip_grad_norm_(self._repa.projector.parameters(), args.max_grad_norm)
        if self._depth_regularizer is not None:
            self._depth_regularizer.release_after_backward(vae)

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
        if not sync_gradients or args.weight_noise_sigma <= 0:
            return
        if self._weight_noise is None:
            from musubi_tuner.training.weight_noise import AdapterWeightNoise, WeightNoiseConfig
            self._weight_noise = AdapterWeightNoise(WeightNoiseConfig(
                args.weight_noise_sigma, args.weight_noise_mode, args.weight_noise_bound_norm,
            ))
        self._weight_noise_logs = self._weight_noise.apply(accelerator.unwrap_model(network).get_trainable_params())

    def extra_step_logs(self, args, logs):
        return dict(self._weight_noise_logs)

    def compute_auxiliary_loss(self, args, accelerator, transformer, network):
        context, self._dop_step_context = self._dop_step_context, None
        if context is None:
            return None, {}

        def restore_swap_layout():
            model = accelerator.unwrap_model(transformer)
            if model.blocks_to_swap:
                model.prepare_block_swap_before_forward()

        return compute_dop_loss(self, args, accelerator, transformer, network, *context,
                                embedding_key="moss_embed", dop_embedding_key="dop_moss_embed",
                                after_teacher=restore_swap_layout)

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        if not args.text_encoder:
            raise ValueError("MOSS sample generation needs the Qwen3.5 text encoder path.")
        prompts = load_prompts(sample_prompts)
        tokenizer = load_tokenizer(args.tokenizer_cache_dir)
        encoder = load_text_encoder(args.text_encoder, device=accelerator.device,
                                    cpu_streaming=args.text_encoder_cpu_streaming, disable_mmap=args.disable_mmap)
        encoded = {}
        for prompt in prompts:
            for text in (prompt.get("prompt", ""), prompt.get("negative_prompt") or ""):
                if text not in encoded:
                    encoded[text] = encode_prompts(encoder, tokenizer, [text], args.moss_max_token_length)[0]
        del encoder
        gc.collect()
        clean_memory_on_device(accelerator.device)
        for prompt in prompts:
            prompt["moss_embed"] = encoded[prompt.get("prompt", "")]
            prompt["negative_moss_embed"] = encoded[prompt.get("negative_prompt") or ""]
        return prompts

    def do_inference(self, accelerator, args, sample_parameter, vae, dit_dtype, transformer,
                     discrete_flow_shift, sample_steps, width, height, frame_count, generator,
                     do_classifier_free_guidance, guidance_scale, cfg_scale, image_path=None, control_video_path=None):
        positive, mask = pad_text_embeds([sample_parameter["moss_embed"]])
        negative, negative_mask = pad_text_embeds([sample_parameter["negative_moss_embed"]])
        latents = generate_latents(
            transformer, positive, mask, negative, negative_mask, height, width, sample_steps,
            cfg_scale if cfg_scale is not None else 3.0,
            float(sample_parameter.get("discrete_flow_shift", args.moss_sample_flow_shift)),
            parse_skip_layers(args.moss_sample_skip_layers), generator.initial_seed() if generator is not None else None,
            accelerator.device,
        )
        vae.to(accelerator.device)
        try:
            with torch.no_grad(), torch.autocast(device_type=accelerator.device.type, enabled=False):
                pixels = vae.decode(latents.to(dtype=torch.float32)).float().clamp(-1, 1).add(1).div(2)
            return pixels.unsqueeze(2).cpu()
        finally:
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)


def moss_setup_parser(parser):
    from musubi_tuner.moss_image.regularization import add_arguments
    add_arguments(parser)
    from musubi_tuner.moss_image.repa import add_arguments as add_repa_arguments
    add_repa_arguments(parser)
    parser.add_argument("--text_encoder")
    parser.add_argument("--tokenizer_cache_dir")
    parser.add_argument("--moss_max_token_length", type=int, default=4096)
    parser.add_argument("--text_encoder_cpu_streaming", action="store_true")
    parser.add_argument("--convrot_int8", action="store_true")
    parser.add_argument("--convrot_bwd_mode", choices=["bf16", "int8"], default="bf16")
    parser.add_argument("--moss_sample_flow_shift", type=float, default=5.0)
    parser.add_argument("--moss_sample_skip_layers", default="22")
    parser.add_argument("--moss_pretraining_schedule", action="store_true")
    parser.add_argument("--moss_logit_mean", type=float, default=-1.09)
    parser.add_argument("--moss_logit_std", type=float, default=0.95)
    parser.add_argument("--moss_timestep_cutoff", type=float, default=0.95)
    parser.add_argument("--moss_timestep_uniform_ratio", type=float, default=0.01)
    parser.add_argument("--disable_mmap", action="store_true")
    return parser


def main():
    parser = moss_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    args.dit_dtype = "bfloat16"
    args.vae_dtype = "float32"
    MossImageNetworkTrainer().train(args)


if __name__ == "__main__":
    main()
