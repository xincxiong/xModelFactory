"""
Shared utilities for training launchers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List


def _require_script_arg(command_name: str) -> tuple[str, List[str]]:
    if len(sys.argv) < 2:
        print(f"Usage: {command_name} <train_script.py> [args...]")
        raise SystemExit(1)
    return sys.argv[1], sys.argv[2:]


def _run(command: List[str], parallel_type: str, prefix: str) -> None:
    os.environ["PARALLEL_TYPE"] = parallel_type
    print(f"[{prefix}] Running: {' '.join(command)}")
    result = subprocess.run(command)
    raise SystemExit(result.returncode)


def _detect_gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0
