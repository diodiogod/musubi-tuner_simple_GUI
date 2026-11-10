"""MOSS-only performance policy; no changes to other model families."""
import logging
import contextlib
import inspect
import os
import sys
from types import MethodType
from pathlib import Path

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def _compiled_adapter_forward(module, inputs):
    if hasattr(module, "_convrot_groupsize"):
        from musubi_tuner.modules.convrot_int8_utils import convrot_int8_linear_forward_patch
        base = convrot_int8_linear_forward_patch(module, inputs)
    else:
        base = F.linear(inputs, module.weight, module.bias)
    for adapter in module._moss_compile_adapters:
        base = adapter(inputs, base_output=base)
    return base


def attach_compile_adapters(model):
    """Make Musubi's monkeypatched adapters visible to the block compiler.

    Musubi owns adapters in a separate network. Dynamo also needs a registered
    path from the compiled block to them. This alias adds no optimizer group
    and leaves the separate network's exported/state keys unchanged. Called
    after the base model's freezing and dtype conversion.
    """
    modules = list(model.modules())
    # ConvRot lazily builds a Hadamard constant. Mutating that global cache
    # while tracing a checkpoint's higher-order graph is an unsafe side effect.
    # Prime every required device/dtype outside the graph, before the first step.
    from musubi_tuner.modules.convrot_int8_kernels import _build_hadamard
    for module in modules:
        size = getattr(module, "_convrot_groupsize", None)
        if size is not None:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                _build_hadamard(size, device=module.weight.device, dtype=dtype)
    for module in modules:
        adapter = getattr(module.forward, "__self__", None)
        chain = []
        while isinstance(adapter, torch.nn.Module) and adapter is not module and hasattr(adapter, "org_forward"):
            if any(adapter is previous for previous in chain):
                raise ValueError("MOSS adapter forwarding contains a cycle.")
            chain.append(adapter)
            adapter = getattr(adapter.org_forward, "__self__", None)
        if chain:
            # Sampling-only adapters wrap the training adapter. Preserve every
            # link and its enabled flag, applying innermost to outermost.
            module.add_module("_moss_compile_adapters", torch.nn.ModuleList(reversed(chain)))
            module.forward = MethodType(_compiled_adapter_forward, module)


def sdpa_context(args, inputs):
    if getattr(args, "moss_sdpa_backend", "default") == "default" or not inputs.is_cuda:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    options = {"set_priority": True} if "set_priority" in inspect.signature(sdpa_kernel).parameters else {}
    return sdpa_kernel(backends, **options)


def pad_context(context, mask, multiple):
    """Reduce compiler shape churn without dropping caption tokens or changing masks."""
    if not multiple:
        return context, mask
    target = ((context.shape[1] + multiple - 1) // multiple) * multiple
    padding = target - context.shape[1]
    return F.pad(context, (0, 0, 0, padding)), F.pad(mask, (0, padding))


def prepare_compile(args, blocks_to_swap):
    if not getattr(args, "compile", False):
        return
    if blocks_to_swap:
        raise ValueError("MOSS compiled blocks require Blocks to swap = 0. Turn Compile off to use swapping.")
    if (getattr(args, "gradient_checkpointing_cpu_offload", False)
            or getattr(args, "cpu_offload_checkpointing", False) or getattr(args, "unsloth_offload_checkpointing", False)):
        raise ValueError("MOSS compiled blocks require ordinary GPU gradient checkpointing; turn checkpoint offloading off.")
    backend = getattr(args, "compile_backend", "inductor")
    if backend == "inductor":
        if tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2]) < (2, 15):
            raise ValueError("MOSS fast compilation needs PyTorch 2.15+. Run tools/bootstrap_environment.py --performance-runtime, or turn Compile off.")
        try:
            import triton  # noqa: F401
        except ImportError as exc:
            raise ValueError("MOSS compilation needs Triton. Install the performance runtime or turn Compile off.") from exc
        if os.name == "nt":
            # Triton Windows ships a C compiler; respect an explicit user CC.
            # Avoid forcing an unrelated ComfyUI installation or global PATH.
            compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
            if compiler.is_file():
                os.environ.setdefault("CC", str(compiler))
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(sys.prefix) / "compile_cache"))
    import torch._dynamo as dynamo
    # Bucketing produces many shapes. Raise the accumulated ceiling too.
    limit = getattr(args, "compile_cache_size_limit", None) or 256
    dynamo.config.cache_size_limit = limit
    if hasattr(dynamo.config, "accumulated_cache_size_limit"):
        dynamo.config.accumulated_cache_size_limit = max(limit, dynamo.config.accumulated_cache_size_limit)
    logger.info("MOSS block compilation: backend=%s, cache limit=%s; first use of a new shape may pause.", backend, limit)
