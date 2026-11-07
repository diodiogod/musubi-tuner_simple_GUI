"""Frozen REPA targets prepared before the DiT, with persistent CPU features."""
from collections import OrderedDict
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open, SafetensorError
from safetensors.torch import save_file
from torch.nn import functional as F
from tqdm import tqdm

from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
CACHE_VERSION = "moss-repa-targets-v1"


def encoder_config(path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    if config.model_type != "dinov2" or config.patch_size != 14:
        raise ValueError("REPA needs a standard patch-14 DINOv2 model folder/ID. "
                         "Other architectures and register-token variants are not supported.")
    return config


def load_encoder(path, config=None):
    from transformers import Dinov2Model
    config = config if config is not None else encoder_config(path)
    # Pin weights to the revision used in the persistent target signature.
    options = {} if Path(path).is_dir() else {"revision": getattr(config, "_commit_hash", None)}
    return Dinov2Model.from_pretrained(path, config=config, **options).eval().requires_grad_(False)


def file_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def latent_key(latent):
    cpu = latent.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(cpu.dtype), tuple(cpu.shape))).encode())
    digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MossRepaTargets:
    def __init__(self, args, training_device, encoder=None):
        self.args = args
        self.training_device = torch.device(training_device)
        self.helper_device = self.training_device if args.moss_repa_device == "training" else torch.device("cpu")
        self.encoder = encoder.eval().requires_grad_(False) if encoder is not None else None
        self.config = encoder.config if encoder is not None else encoder_config(args.moss_repa_encoder)
        self.feature_dim = self.config.hidden_size
        self.cache = OrderedDict()
        self.paths = None  # None permits the standalone, on-demand compatibility path.
        self.signature = None
        self._injected_encoder = encoder is not None

    def _signature(self):
        path = Path(self.args.moss_repa_encoder)
        if self._injected_encoder:
            digest = hashlib.sha256()
            for key, value in self.encoder.state_dict().items():
                digest.update(key.encode())
                digest.update(latent_key(value).encode())
            encoder = {"injected_weights": digest.hexdigest()}
        elif path.is_dir():
            files = {p for p in path.iterdir() if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"}}
            for index in path.glob("*.index.json"):
                mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
                files.update(path / name for name in mapping.values())
            # Do not include recursively generated target files if the user
            # chooses the local teacher folder as the optional cache location.
            encoder = {"local": [file_identity(p) for p in sorted(files)]}
        else:
            revision = getattr(self.config, "_commit_hash", None)
            if not revision:
                raise ValueError("Could not identify the DINOv2 revision for safe REPA feature reuse.")
            encoder = {"repo": self.args.moss_repa_encoder, "revision": revision}
        identity = {"version": CACHE_VERSION, "encoder": encoder,
                    "encoder_config": self.config.to_dict(), "vae": file_identity(self.args.vae),
                    "image_size": self.args.moss_repa_image_size, "dtype": "float32"}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()

    def _read(self, path, key):
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("signature") != self.signature or metadata.get("latent") != key:
                    return None
                features = handle.get_tensor("features")
                expected = (1, (self.args.moss_repa_image_size // 14) ** 2, self.feature_dim)
                if tuple(features.shape) != expected or features.dtype != torch.float32 or not torch.isfinite(features).all():
                    return None
                # Do not retain mmap handles for every training image on Windows.
                return features.clone()
        except (OSError, SafetensorError, ValueError, KeyError):
            return None

    def _write(self, path, key, features):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".repa-", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            save_file({"features": features.contiguous()}, temporary,
                      metadata={"signature": self.signature, "latent": key, "version": CACHE_VERSION})
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _remember(self, key, features):
        if self.args.moss_repa_cache_size:
            self.cache[key] = features
            self.cache.move_to_end(key)
            while len(self.cache) > self.args.moss_repa_cache_size:
                self.cache.popitem(last=False)

    def release_helpers(self, vae):
        vae.to("cpu")
        if self.encoder is not None:
            self.encoder.to("cpu")
        if self.helper_device.type == "cuda":
            clean_memory_on_device(self.helper_device)

    @torch.no_grad()
    def _encode(self, vae, sample, preparing=False):
        vae.to(self.helper_device).eval().requires_grad_(False)
        with torch.autocast(device_type=self.helper_device.type, enabled=False):
            pixels = vae.decode(sample.unsqueeze(0).squeeze(2).to(self.helper_device, dtype=torch.float32))
            pixels = pixels.float().clamp(-1, 1).add(1).div(2)
            pixels = F.interpolate(pixels, size=(self.args.moss_repa_image_size,) * 2,
                                   mode="bilinear", align_corners=False)
        if not preparing:
            vae.to("cpu")
        if self.encoder is None:
            self.encoder = load_encoder(self.args.moss_repa_encoder, self.config)
        self.encoder.to(device=self.helper_device, dtype=torch.float32)
        with torch.autocast(device_type=self.helper_device.type, enabled=False):
            mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            output = self.encoder(pixel_values=(pixels - mean) / std,
                                  interpolate_pos_encoding=True, return_dict=True)
            return output.last_hidden_state[:, 1:].float().cpu().contiguous()

    @torch.no_grad()
    def prepare(self, dataset_group, vae):
        """Resolve every used latent exactly once, while no DiT is resident."""
        self.signature = self._signature()
        items = {}
        for dataset in dataset_group.datasets:
            for bucket in dataset.batch_manager.buckets.values():
                for item in bucket:
                    path = Path(item.latent_cache_path).resolve()
                    items.setdefault(path, Path(dataset.cache_directory))
        self.paths = {}
        generated = reused = 0
        devices = [self.training_device.index or 0] if self.training_device.type == "cuda" else []
        # A cold encoder load must not change training randomness relative to a
        # warm cache or a resume. Helpers themselves are deterministic and frozen.
        try:
            with torch.random.fork_rng(devices=devices):
                for path, directory in tqdm(items.items(), desc="REPA feature cache", total=len(items)):
                    with safe_open(str(path), framework="pt", device="cpu") as handle:
                        keys = [key for key in handle.keys()
                                if key.startswith("latents_") and not key.startswith("latents_control_")]
                        if len(keys) != 1:
                            raise ValueError(f"REPA expected one MOSS training latent in {path}")
                        sample = handle.get_tensor(keys[0])
                    if sample.ndim != 4 or tuple(sample.shape[:2]) != (128, 1):
                        raise ValueError(f"REPA requires packed MOSS image latents [128,1,H,W]: {path}")
                    key = latent_key(sample)
                    if key in self.paths:
                        continue  # Repeats and identical images need no additional work.
                    root = str(getattr(self.args, "moss_repa_cache_dir", "") or "").strip()
                    root = Path(root).expanduser() if root else directory / "moss_repa_features"
                    destination = root / self.signature / (key + ".safetensors")
                    result = self._read(destination, key)
                    if result is None:
                        result = self._encode(vae, sample, preparing=True)
                        self._write(destination, key, result)
                        generated += 1
                    else:
                        reused += 1
                    self.paths[key] = destination
                    self._remember(key, result)
        finally:
            self.release_helpers(vae)
            if not self._injected_encoder:
                self.encoder = None  # DINO is not needed during training or previews.
        logger.info("REPA targets ready before DiT loading: %s generated, %s reused; "
                    "training reads saved features without REPA VAE decoding or teacher transfers.", generated, reused)
        return {"generated": generated, "reused": reused}

    @torch.no_grad()
    def targets(self, vae, latents):
        results = []
        computed = False
        try:
            for sample in latents:
                key = latent_key(sample)
                if key in self.cache:
                    result = self.cache[key]
                    self.cache.move_to_end(key)
                elif self.paths is not None:
                    path = self.paths.get(key)
                    result = self._read(path, key) if path is not None else None
                    if result is None:
                        raise RuntimeError("REPA feature cache is missing, stale or damaged. Restart the run to "
                                           "prepare it before training; the trainer will not decode targets mid-step.")
                    self._remember(key, result)
                else:
                    computed = True
                    result = self._encode(vae, sample)
                    self._remember(key, result)
                results.append(result)
        finally:
            if computed:
                self.release_helpers(vae)
        # Cache hits never move the VAE: honor depth helper residency and avoid
        # allocator flushes while the training model is live.
        return torch.cat(results).to(self.training_device)
