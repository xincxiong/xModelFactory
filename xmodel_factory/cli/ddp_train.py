"""
Launch training with PyTorch DDP.
"""

from __future__ import annotations

from ._launcher import _detect_gpu_count, _require_script_arg, _run


def main() -> None:
    run_file, extra_args = _require_script_arg("ddp_train")
    gpu_count = max(1, _detect_gpu_count())
    _run(
        ["torchrun", "--standalone", "--nproc_per_node", str(gpu_count), run_file, *extra_args],
        "ddp",
        "ddp_train",
    )


if __name__ == "__main__":
    main()
