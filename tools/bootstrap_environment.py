#!/usr/bin/env python3
"""Create or validate the shared Musubi GUI environment.

This intentionally keeps launching cheap: a healthy environment with a current
fingerprint performs only a small import check. Installation output is written
to logs/setup.log so launcher errors remain readable for nontechnical users.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "venv"
PYPROJECT = ROOT / "pyproject.toml"
MARKER = VENV / ".musubi_setup.json"
LOG = ROOT / "logs" / "setup.log"
BOOTSTRAP_VERSION = "1"
REQUIRED_IMPORTS = ("torch", "accelerate", "bitsandbytes", "transformers", "diffusers", "safetensors")
CUDA_MINIMUMS = {
    "cu124": ("torch>=2.5.1", "torchvision>=0.20.1"),
    "cu128": ("torch>=2.7.1", "torchvision>=0.22.1"),
    "cu130": ("torch>=2.9.1", "torchvision>=0.24.1"),
    "cu132": ("torch>=2.12.0", "torchvision>=0.27.0"),
}


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(PYPROJECT.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(BOOTSTRAP_VERSION.encode())
    return digest.hexdigest()


def valid_python_version() -> bool:
    return (3, 10) <= sys.version_info[:2] < (3, 13)


def run_logged(command: Sequence[str]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}. See {LOG}")


def environment_healthy(python: Path) -> bool:
    imports = ", ".join(REQUIRED_IMPORTS)
    check = subprocess.run(
        [str(python), "-c", f"import {imports}; assert torch.__version__"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check.returncode:
        return False
    pip_check = subprocess.run(
        [str(python), "-m", "pip", "check"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return pip_check.returncode == 0


def environment_available(python: Path) -> bool:
    """Fast repeat-launch check that avoids importing the large Torch stack."""
    modules = repr(REQUIRED_IMPORTS)
    expression = f"import importlib.util; assert all(importlib.util.find_spec(name) for name in {modules})"
    return subprocess.run(
        [str(python), "-c", expression],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def installed_torch_usable(python: Path, nvidia_available: bool) -> bool:
    expression = "import torch; " + ("assert torch.cuda.is_available()" if nvidia_available else "assert torch.__version__")
    return subprocess.run(
        [str(python), "-c", expression],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def detected_driver_cuda() -> tuple[bool, tuple[int, int] | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False, None
    result = subprocess.run([executable], capture_output=True, text=True, errors="replace", check=False)
    if result.returncode:
        return False, None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    return True, (int(match.group(1)), int(match.group(2))) if match else None


def choose_torch_channel(driver_cuda: tuple[int, int] | None) -> str:
    override = os.environ.get("MUSUBI_CUDA", "").strip().lower()
    if override:
        if override not in {*CUDA_MINIMUMS, "cpu"}:
            raise ValueError(f"Unsupported MUSUBI_CUDA={override!r}; choose cu124, cu128, cu130, cu132, or cpu.")
        return override
    if driver_cuda and driver_cuda >= (12, 8):
        return "cu128"
    return "cu124"


def install_torch(python: Path, channel: str) -> None:
    if channel == "cpu":
        packages = ("torch>=2.5.1", "torchvision>=0.20.1")
        index = "https://download.pytorch.org/whl/cpu"
    else:
        packages = CUDA_MINIMUMS[channel]
        index = f"https://download.pytorch.org/whl/{channel}"
    print(f"Installing the PyTorch {channel} build. This can take several minutes...")
    run_logged([str(python), "-m", "pip", "install", "--upgrade", *packages, "--index-url", index])


def write_marker(state: str, channel: str | None = None) -> None:
    payload = {"fingerprint": fingerprint(), "state": state, "torch_channel": channel}
    MARKER.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_marker() -> dict:
    try:
        return json.loads(MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the shared Musubi GUI Python environment.")
    parser.add_argument("--force-repair", action="store_true", help="reinstall project dependencies")
    parser.add_argument("--refresh-torch", action="store_true", help="reinstall the selected PyTorch CUDA build")
    args = parser.parse_args()

    if not PYPROJECT.is_file():
        print(f"ERROR: {PYPROJECT} was not found.", file=sys.stderr)
        return 1
    if not valid_python_version():
        print(
            f"ERROR: Python {sys.version_info.major}.{sys.version_info.minor} is unsupported. "
            "Install Python 3.10, 3.11, or 3.12.",
            file=sys.stderr,
        )
        return 1

    python = venv_python()
    created = not python.is_file()
    if created:
        print("Creating the shared Musubi virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT, check=True)
        python = venv_python()

    marker = read_marker()
    current = marker.get("fingerprint") == fingerprint()
    if current and environment_available(python) and not args.force_repair and not args.refresh_torch:
        print("Musubi environment is ready.")
        return 0
    healthy = environment_healthy(python)
    if healthy and not marker and not args.force_repair and not args.refresh_torch:
        # Preserve a manually prepared working environment on the first run of
        # the smart launcher instead of unexpectedly replacing its Torch stack.
        write_marker("adopted")
        print("Existing Musubi environment verified and adopted.")
        return 0

    nvidia_available, driver_cuda = detected_driver_cuda()
    channel = choose_torch_channel(driver_cuda) if nvidia_available else "cpu"
    refresh_torch = args.refresh_torch or not installed_torch_usable(python, nvidia_available)

    try:
        print(f"Preparing dependencies. Detailed output: {LOG}")
        run_logged([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        if refresh_torch:
            install_torch(python, channel)
        print("Installing or updating Musubi GUI dependencies...")
        run_logged([str(python), "-m", "pip", "install", "--upgrade", "-e", "."])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not environment_healthy(python):
        print(f"ERROR: Dependency verification failed. See {LOG}", file=sys.stderr)
        return 1
    write_marker("installed", channel)
    print("Musubi environment installed and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
