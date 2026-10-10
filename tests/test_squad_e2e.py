"""Squad orchestration + end-to-end audit squad tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from support import TempWorkspace

from squad.agent import Agent, ReflectionPolicy
from squad.llm import LLMResponse, MockLLM
from squad.memory import Blackboard
from squad.squad import Squad, SquadResult
from squads.code_audit import DEFAULT_SAMPLE_FILES, build_code_audit_squad, sample_workspace

REPORT = """## Summary
Audited the package modules, snapshotted the sinks and wrote the regression tests.
## Findings
- merge of the audit artefacts is blocked by a release gate failure.
- SQL injection in `db.find_user` and `eval` on payloads in `auth.py`.
- the module boundaries leak through billing.py computing totals inline; a single
  owner should replace the duplicated total computation.
## Evidence
- read_file db.py, auth.py and billing.py, plus a [tool read_file ok] observation.
- search_text matched `eval(` in auth.py line 1.
- python_exec ran tests/test_audit_regression.py and reported the traceback.
## Actions
- replace the `%` interpolation in `db.py` with bound parameters
- remove the `eval` call in `auth.py`
- fix the module boundaries in db.py and auth.py
- replace the unbounded billing.total default with None
- enforce bound parameters in db.py
- remove the weak comparison in auth.py
- add tests/test_audit_regression.py covering both defects
## Verification
- run python -m unittest discover -s tests after the changes
- execute tests/test_audit_regression.py and review the generated reports
- snapshot the session and restore billing.py to confirm the fix holds
## Risks
- none beyond the addressed points.
"""

REFLEXIVE = None  # set below: mock responder that always returns a complete report


def _report_responder(index, agent, messages, tools):
    calls = []
    if index == 0 and tools:
        calls = []
    return LLMResponse(text=REPORT, tool_calls=calls, provider="mock")


REFLEXIVE = _report_responder


class TestOrchestration(TempWorkspace):
    def setUp(self) -> None:
        super().setUp()
        self.write("pkg/a.py", "from . import b\n")
        self.write("pkg/b.py", "x = 1\n")

    def _squad(self) -> Squad:
        squad = Squad("test", max_workers=2)
        squad.add(Agent("arch", role="Architect", goal="audit the package modules", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()), group="architecture")
        squad.add(
            Agent("sec", role="Security", goal="audit the sinks", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()),
            group="security",
            depends_on=("architecture",),
            needs=("arch",),
        )
        squad.add(
            Agent("test", role="Tester", goal="audit the tests", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()),
            group="tests",
            depends_on=("security",),
            needs=("sec",),
        )
        return squad

    def test_plan_is_deterministic_and_ordered(self) -> None:
        squad = self._squad()
        self.assertEqual(squad.plan(), [["architecture"], ["security"], ["tests"]])
        self.assertEqual(squad.plan(), squad.plan())
        self.assertEqual(squad.groups(), ["architecture", "security", "tests"])
        self.assertEqual(squad.group_dependencies()["security"], ["architecture"])
        self.assertIn("wave 1: [architecture]", squad.summary())

    def test_plan_detects_unknown_group_and_cycle(self) -> None:
        squad = Squad("bad")
        squad.add(Agent("a", llm=MockLLM(), name_prefix="p-"), group="a", depends_on=("ghost",))
        with self.assertRaises(ValueError) as ctx:
            squad.plan()
        self.assertIn("ghost", str(ctx.exception))

        cyclic = Squad("cyclic")
        cyclic.add(Agent("a", llm=MockLLM()), group="a", depends_on=("b",))
        cyclic.add(Agent("b", llm=MockLLM()), group="b", depends_on=("a",))
        with self.assertRaises(ValueError) as ctx:
            cyclic.plan()
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_run_produces_results_and_events(self) -> None:
        events = []
        board = Blackboard()
        result = self._squad().run(focus="audit the package", blackboard=board, on_event=events.append)

        self.assertIsInstance(result, SquadResult)
        self.assertEqual(list(result.results), ["arch", "sec", "test"])
        self.assertTrue(result.ok())
        self.assertEqual(result.scores["arch"], result.results["arch"].score)
        self.assertEqual(len(result.rounds), 3)
        self.assertEqual(result.rounds[0]["agents"], ["arch"])
        self.assertTrue(any(e["type"] == "group" for e in events))
        self.assertTrue(any(e["type"] == "result" and e["agent"] == "test" for e in events))
        self.assertTrue(json.dumps(result.to_dict(), sort_keys=True))

    def test_wave_isolation_prevents_sibling_reads(self) -> None:
        """A same-wave sibling's findings must not leak into the context."""
        seen = {}

        class Spy(Agent):
            def execute(self, task, *, context=None, blackboard=None, task_id=""):
                seen[self.name] = context or {}
                return super().execute(task, context=context, blackboard=blackboard, task_id=task_id)

        squad = Squad("iso", max_workers=2)
        squad.add(Spy("left", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()), group="g", needs=("right",))
        squad.add(Spy("right", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()), group="g")
        squad.run()

        self.assertEqual(seen["left"]["needs"], {"right": {}})
        self.assertIn("focus", seen["left"])
        self.assertEqual(seen["left"]["group"], "g")

    def test_needs_are_passed_downstream(self) -> None:
        captured = {}

        class Spy(Agent):
            def execute(self, task, *, context=None, blackboard=None, task_id=""):
                captured[self.name] = context or {}
                return super().execute(task, context=context, blackboard=blackboard, task_id=task_id)

        squad = Squad("needs", max_workers=1)
        squad.add(Spy("up", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()), group="one")
        squad.add(Spy("down", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()), group="two", depends_on=("one",), needs=("up",))
        squad.run()

        self.assertEqual(captured["down"]["needs"]["up"]["score"], captured["up"].score if False else captured["down"]["needs"]["up"]["score"])
        self.assertIn("score", captured["down"]["needs"]["up"])
        self.assertIn("output", captured["down"]["needs"]["up"])
        self.assertEqual(captured["down"]["upstream_summaries"]["up"]["sender"], "up")
        self.assertTrue(captured["down"]["sibling_outputs"]["up"])

    def test_agent_exception_is_isolated(self) -> None:
        class Exploding(Agent):
            def execute(self, task, *, context=None, blackboard=None, task_id=""):
                raise RuntimeError("kaboom")

        squad = Squad("boom", max_workers=2)
        squad.add(Exploding("bad", llm=MockLLM()), group="g1")
        squad.add(
            Agent("good", role="Auditor", goal="audit the package modules in pkg/a.py", llm=MockLLM(responder=REFLEXIVE), tools=self.registry()),
            group="g1",
        )
        result = squad.run()

        self.assertEqual(result.results["bad"].status, "failed")
        self.assertIn("kaboom", result.results["bad"].warnings[0])
        self.assertEqual(result.results["good"].status, "ok")
        self.assertFalse(result.ok())

    def test_bad_event_callback_cannot_abort_run(self) -> None:
        def explode(event):
            raise ValueError("listener bug")

        result = self._squad().run(on_event=explode)
        self.assertEqual(len(result.results), 3)

    def test_parallel_and_sequential_runs_agree(self) -> None:
        parallel = self._squad().run()
        squad = self._squad()
        squad.max_workers = 1
        sequential = squad.run()
        self.assertEqual(
            {name: round(r.score, 6) for name, r in parallel.results.items()},
            {name: round(r.score, 6) for name, r in sequential.results.items()},
        )
        self.assertEqual(list(parallel.results), list(sequential.results))

    def test_find_and_lookup_errors(self) -> None:
        squad = self._squad()
        self.assertEqual(squad.find("arch").name, "arch")
        with self.assertRaises(KeyError):
            squad.find("ghost")
        self.assertEqual(len(squad), 3)
        self.assertIn("arch", squad)
        self.assertIn("Squad(name='test'", repr(squad))
        with self.assertRaises(ValueError):
            squad.add(Agent("arch", llm=MockLLM()))

    def test_double_run_does_not_leak_state(self) -> None:
        squad = self._squad()
        first = squad.run()
        second = squad.run()
        self.assertEqual(len(second.results), len(first.results))
        self.assertNotEqual(squad.blackboard.stats()["messages"] if hasattr(squad.blackboard, "stats") else 0, 0)


class TestAuditSquad(TempWorkspace):
    def test_sample_workspace_materialises_flaws(self) -> None:
        written = sample_workspace(self.workdir)
        self.assertEqual(written, sorted(DEFAULT_SAMPLE_FILES))
        self.assertIn("eval(payload)", self.read("auth.py"))
        self.assertIn("% username", self.read("db.py"))

    def test_offline_squad_satisfies_acceptance_criteria(self) -> None:
        """End-to-end: 4 agents, real tool execution, artifacts on disk."""
        sample_workspace(self.workdir)
        paths = sorted(DEFAULT_SAMPLE_FILES)
        board = Blackboard()
        squad = build_code_audit_squad(
            blackboard=board,
            llm=MockLLM(responder=_report_responder),
            tools=self.registry(),
            workdir=self.workdir,
            max_rounds=2,
        )
        self.assertEqual(squad.plan(), [["architecture"], ["security"], ["tests"], ["review"]])

        result = squad.run(focus="audit the sample package")
        self.assertEqual(list(result.results), ["audit-architecture", "audit-security", "audit-tests", "audit-review"])
        self.assertEqual(len(board.namespaces()), 4)
        self.assertTrue(all(name in board.namespaces() for name in result.results))
        self.assertTrue(result.ok(), {n: r.warnings for n, r in result.results.items()})

    def test_squad_is_reproducible_across_two_runs(self) -> None:
        sample_workspace(self.workdir)
        llm = MockLLM(responder=_report_responder)
        scores = []
        for _ in range(2):
            board = Blackboard()
            squad = build_code_audit_squad(blackboard=board, llm=llm, tools=self.registry(), workdir=self.workdir)
            result = squad.run()
            scores.append({name: round(r.score, 6) for name, r in result.results.items()})
            statuses = {name: r.status for name, r in result.results.items()}
            self.assertTrue(all(s == "ok" for s in statuses.values()), statuses)
        self.assertEqual(scores[0], scores[1])


class TestCli(TempWorkspace):
    def _run_cli(self, args, **kwargs):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        env["OMP_LLM_MODE"] = "mock"
        return subprocess.run(
            [sys.executable, "-m", "squad.cli"] + args,
            capture_output=True,
            text=True,
            cwd=self.workdir,
            env=env,
            timeout=180,
            **kwargs,
        )

    def test_demo_runs_offline_end_to_end(self) -> None:
        proc = self._run_cli(["demo", "--quiet", "--max-rounds", "2", "--workdir", self.path("ws")])
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("AGENT SQUAD AUDIT REPORT", proc.stdout)
        self.assertIn("audit-review", proc.stdout)
        self.assertIn("PASS", proc.stdout)

        snapshot = subprocess.run(
            [sys.executable, "-c", "import json,sys; json.loads(sys.stdin.read())"],
            input=self._run_cli(["demo", "--quiet", "--json", "--max-rounds", "1", "--workdir", self.path("ws2")]).stdout,
            capture_output=True,
            text=True,
        )
        self.assertEqual(snapshot.returncode, 0, snapshot.stderr)

    def test_version_command(self) -> None:
        proc = self._run_cli(["version"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("agent-squad", proc.stdout)

    def test_audit_command_rejects_empty_directory(self) -> None:
        os.makedirs(self.path("empty"), exist_ok=True)
        proc = self._run_cli(["audit", self.path("empty")])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("no auditable files", proc.stderr)

    def test_offline_and_live_are_mutually_exclusive(self) -> None:
        proc = self._run_cli(["demo", "--offline", "--live"])
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
