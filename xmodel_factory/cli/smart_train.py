"""
Auto-select the best available training backend.
"""

from __future__ import annotations

import sys

from ._launcher import _detect_gpu_count, _require_script_arg, _run


def main() -> None:
    run_file, extra_args = _require_script_arg("smart_train")

    try:
        import deepspeed  # noqa: F401

        has_deepspeed = True
    except ImportError:
        has_deepspeed = False

    gpu_count = _detect_gpu_count()

    if has_deepspeed:
        print(f"[smart_train] Using DeepSpeed (detected {gpu_count} GPUs)")
        _run(["deepspeed", run_file, *extra_args], "ds", "smart_train")
    if gpu_count > 1:
        print(f"[smart_train] Using DDP with {gpu_count} GPUs")
        _run(
            ["torchrun", "--standalone", "--nproc_per_node", str(gpu_count), run_file, *extra_args],
            "ddp",
            "smart_train",
        )

    print("[smart_train] Using single device training")
    _run([sys.executable, run_file, *extra_args], "none", "smart_train")


if __name__ == "__main__":
    main()
