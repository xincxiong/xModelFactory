"""
Launch training with DeepSpeed.
"""

from __future__ import annotations

from ._launcher import _require_script_arg, _run


def main() -> None:
    run_file, extra_args = _require_script_arg("ds_train")
    _run(["deepspeed", run_file, *extra_args], "ds", "ds_train")


if __name__ == "__main__":
    main()
