"""Shared test fixtures: workspace helpers and deterministic provider builders."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from squad.llm import MockLLM  # noqa: E402
from squad.tools import build_default_registry  # noqa: E402


class TempWorkspace(unittest.TestCase):
    """Base class giving each test an isolated sandbox directory."""

    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp(prefix="squad-test-")
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)

    def path(self, *parts: str) -> str:
        return os.path.join(self.workdir, *parts)

    def write(self, rel: str, content: str) -> str:
        target = self.path(rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)
        return target

    def read(self, rel: str) -> str:
        with open(self.path(rel), "r", encoding="utf-8") as handle:
            return handle.read()

    def registry(self, **kwargs):
        return build_default_registry(workdir=self.workdir, **kwargs)

    def mock(self, responder=None) -> MockLLM:
        return MockLLM(responder=responder)
