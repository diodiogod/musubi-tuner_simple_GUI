"""Experimental graph-boundary residency changes over Musubi H2D streaming.

Selected only by the explicit H3 automatic-mode factory branch.
Call ``reconfigure`` only after every backward belonging to a microbatch completes.
"""

import logging

import torch
from torch import nn

from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader


class AutomaticLoRAStreamOffloader(LoRAStreamOffloader):
    def __init__(self, block_type, blocks, num_blocks, supports_backward, device,
                 use_pinned_memory=False, debug=False, swap_tensor_selector=None):
        # Master all eligible weights before promoting any block. Register hooks
        # on every block through the override below, not the initial fixed layout.
        super().__init__(block_type, blocks, num_blocks, num_blocks,
                         supports_backward, device, ring_size=2,
                         use_pinned_memory=use_pinned_memory, debug=debug,
                         swap_tensor_selector=swap_tensor_selector)
        self.resident_flat = {}
        self.resident_param = {}
        self._automatic_initialized = False
        self._microbatch_active = False

    def begin_microbatch(self):
        if self._microbatch_active:
            raise RuntimeError("Automatic swap microbatch is already active")
        self._microbatch_active = True

    def wait_for_block(self, block_idx):
        with torch.cuda.device(self.device):
            return super().wait_for_block(block_idx)

    def submit_move_blocks_forward(self, blocks, block_idx):
        with torch.cuda.device(self.device):
            return super().submit_move_blocks_forward(blocks, block_idx)

    def end_microbatch(self):
        if not self._microbatch_active:
            raise RuntimeError("Automatic swap microbatch was not started")
        self._microbatch_active = False

    def _create_backward_hook(self, block_index):
        def hook(module, grad_input, grad_output):
            with torch.cuda.device(self.device):
                if self.is_stream[block_index]:
                    rank = self.rank[block_index]
                    self.free_event[rank % self.B] = torch.cuda.current_stream(self.device).record_event()
                    if rank >= self.B:
                        self._load(rank - self.B, (rank - self.B) % self.B, "bwd")
                if block_index > 0 and self.is_stream[block_index - 1]:
                    self._wait_ctx = "bwd"
                    try:
                        self.wait_for_block(block_index - 1)
                    finally:
                        self._wait_ctx = "fwd"
        return hook

    def prepare_block_devices_before_forward(self, blocks):
        with torch.cuda.device(self.device):
            if not self._automatic_initialized:
                template = None
                total_bytes = 0
                for i in range(self.num_blocks):
                    weights = [getattr(m, name) for m, name, _ in self._jobs(i)]
                    signature = [(tuple(w.shape), w.dtype) for w in weights]
                    if template is not None and signature != template:
                        raise ValueError("Automatic streaming requires homogeneous frozen block layouts")
                    template = signature
                    total_bytes += self._compute_layout(weights)[1]
                logging.getLogger(__name__).info(
                    "Automatic swap CPU masters: %.2f GiB (%s); excludes adapters and retained buffers",
                    total_bytes / 1024**3, "pinned" if self.use_pinned_memory else "pageable")
                super().prepare_block_devices_before_forward(blocks)
                self._automatic_initialized = True
            else:
                self.copier.sync()
                torch.cuda.synchronize(self.device)
                for block, tensors in self.resident_param.items():
                    self._bind(block, tensors)
                super().prepare_block_devices_before_forward(blocks)

    def set_forward_only(self, forward_only):
        if self._microbatch_active:
            raise RuntimeError("Cannot enter preview with a live automatic-swap microbatch")
        if self._automatic_initialized:
            # Preview is not a training bucket. Release dedicated residency and
            # let the next measured training boundary choose its plan afresh.
            self.reconfigure(self.num_blocks)
        super().set_forward_only(forward_only)

    def reconfigure(self, streamed):
        if self._microbatch_active:
            raise RuntimeError("Cannot change block residency during a live microbatch")
        if not self._automatic_initialized:
            raise RuntimeError("Prepare CPU masters before changing residency")
        if not 2 <= streamed <= self.num_blocks:
            raise ValueError("Automatic streaming count must be between 2 and the block count")
        if streamed == self.S:
            return
        # Nested sets preserve existing dedicated allocations. Reverse ordering
        # keeps a simple deterministic contiguous streaming prefix initially.
        wanted = set(range(streamed, self.num_blocks))
        with torch.cuda.device(self.device):
            self.copier.sync()
            torch.cuda.synchronize(self.device)
            self.copier.reset()
            for block in range(self.num_blocks):
                self._bind(block, self.cpu_master[block])
            self.in_slot = [None] * self.B
            self.free_event = [None] * self.B
            # Demote before any promotion. Only frozen selected tensors move;
            # adapter parameters, accumulated gradients and scales stay intact.
            for block in set(self.resident_flat) - wanted:
                del self.resident_param[block]
                del self.resident_flat[block]
            try:
                for block in sorted(wanted - set(self.resident_flat)):
                    flat = self.cpu_flat[block].to(self.device)
                    views = self._flat_views(flat, self.cpu_master[block], self._layout)
                    parameters = [
                        nn.Parameter(view, requires_grad=False) if is_param else view
                        for (_, _, is_param), view in zip(self._jobs(block), views)
                    ]
                    self.resident_flat[block] = flat
                    self.resident_param[block] = parameters
            finally:
                # Even failed allocation leaves a coherent, graph-free layout.
                # Propagate errors; no replay of partially executed training.
                self.stream_idx = [b for b in range(self.num_blocks) if b not in self.resident_param]
                self.rank = {b: k for k, b in enumerate(self.stream_idx)}
                self.is_stream = [b in self.rank for b in range(self.num_blocks)]
                self.S = self.blocks_to_swap = len(self.stream_idx)
                for block, tensors in self.resident_param.items():
                    self._bind(block, tensors)
                for rank in range(self.B):
                    self._load(rank, rank)
                self.copier.sync()
                torch.cuda.synchronize(self.device)
