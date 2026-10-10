"""Test package bootstrap: put ``src/`` and ``tests/`` on ``sys.path``."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
