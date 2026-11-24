"""Optional per-image loss capture. Transfer detached batches together to CPU."""

import json
import math
import time

import torch


class SampleLossLog:
    def __init__(self, path):
        self.path = path
        self.pending = []
        self.last_flush = time.monotonic()

    def add(self, paths, times, losses, step, epoch, shape):
        self.pending.append((paths, torch.stack((times.detach().float(), losses.detach())), step, epoch, shape))
        if len(self.pending) >= 32 or time.monotonic() - self.last_flush >= 5:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        values = torch.cat([row[1] for row in self.pending], dim=1).cpu().tolist()
        cursor = 0
        with self.path.open("a", encoding="utf-8") as handle:
            for paths, _, step, epoch, shape in self.pending:
                for path in paths:
                    sigma, loss = values[0][cursor], values[1][cursor]
                    cursor += 1
                    if math.isfinite(sigma) and math.isfinite(loss):
                        handle.write(json.dumps({"image": path, "timestep": sigma, "loss": loss, "step": step,
                                                 "epoch": epoch, "grid": shape}) + "\n")
        self.pending.clear()
        self.last_flush = time.monotonic()
