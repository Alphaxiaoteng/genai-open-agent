"""Concurrency + determinism stress tests (the squad's reproducibility claim)."""

from __future__ import annotations

import json
import os
import threading
import unittest

from support import TempWorkspace

from squad.agent import Agent
from squad.llm import LLMResponse, MockLLM
from squad.memory import Blackboard
from squad.squad import Squad

REPORT = """## Summary
Touched `mod.py` while auditing.
## Findings
- one structural risk in `mod.py`.
## Evidence
- read_file mod.py and a [tool read_file ok] observation.
## Actions
- split the module
## Verification
- run python -m unittest
## Risks
- none beyond the addressed points.
"""


def responder(index, agent, messages, tools):
    return LLMResponse(text=REPORT, provider="mock")


class TestConcurrency(TempWorkspace):
    def test_wide_fanout_with_shared_blackboard(self) -> None:
        """16 agents in one wave, all publishing to one blackboard."""
        self.write("mod.py", "x = 1\n")
        board = Blackboard()
        squad = Squad("wide", max_workers=8)
        for index in range(16):
            squad.add(
                Agent("w%02d" % index, role="Worker", goal="audit mod.py", llm=MockLLM(responder=responder), tools=self.registry()),
                group="g",
            )
        result = squad.run(blackboard=board)

        self.assertEqual(len(result.results), 16)
        self.assertTrue(result.ok())
        self.assertEqual(len(board.namespaces()), 16)
        # 16 assignments + 16 summaries.
        self.assertEqual(board.stats()["messages"], 32)

    def test_tool_registry_under_thread_contention(self) -> None:
        self.write("mod.py", "y = 2\n")
        registry = self.registry()
        errors = []

        def worker() -> None:
            try:
                for _ in range(25):
                    result = registry.call("read_file", {"path": "mod.py"})
                    if not result.ok:
                        errors.append(result.error)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(registry.audit_log()), 8 * 25)

    def test_repeated_runs_are_byte_stable(self) -> None:
        self.write("mod.py", "z = 3\n")
        snapshots = []
        for _ in range(3):
            board = Blackboard()
            squad = Squad("stable", max_workers=4)
            for name in ("a", "b", "c"):
                squad.add(
                    Agent(name, role="W", goal="audit mod.py", llm=MockLLM(responder=responder), tools=self.registry()),
                    group="g",
                )
            result = squad.run(blackboard=board)
            snapshots.append(
                json.dumps(
                    {
                        name: {"status": res.status, "score": round(res.score, 6), "revisions": res.revisions}
                        for name, res in result.results.items()
                    },
                    sort_keys=True,
                )
            )
        self.assertEqual(len(set(snapshots)), 1, snapshots)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
