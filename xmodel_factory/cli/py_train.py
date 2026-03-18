"""
Launch training on a single process.
"""

from __future__ import annotations

import sys

from ._launcher import _require_script_arg, _run


def main() -> None:
    run_file, extra_args = _require_script_arg("py_train")
    _run([sys.executable, run_file, *extra_args], "none", "py_train")


if __name__ == "__main__":
    main()
