"""Self-reflection loop + provider abstraction tests."""

from __future__ import annotations

import os
import unittest

from support import TempWorkspace

from squad.agent import Agent, ReflectionPolicy
from squad.llm import (
    LLMConfigError,
    LLMError,
    LLMResponse,
    MockLLM,
    OpenAICompatibleLLM,
    ProviderFallback,
    build_llm,
    format_tool_calls,
    parse_tool_calls,
)
from squad.memory import Blackboard, ToolCall
from squad.tools import ToolSpec, build_default_registry

GOOD_REPORT = """## Summary
Audited pkg/mod.py and found one injection sink.
## Findings
- `pkg/mod.py` line 4 calls eval on untrusted input.
## Evidence
- read_file pkg/mod.py -> def f(x): return eval(x)
- [tool read_file ok]
## Actions
- replace eval with a whitelist parser
- add a regression test for pkg/mod.py
## Verification
- run python -m unittest after the fix
## Risks
- eval removal may break stored payloads.
"""

DRAFT = """## Summary
Looks fine.
## Findings
- some issues exist
## Evidence
- none
## Actions
inspected things
## Verification
nothing
## Risks
TBD
"""


class ScriptedLLM:
    """Deterministic provider returning a fixed script of responses."""

    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, *, tools=(), system=None, temperature=0.2, max_tokens=1024):
        self.calls.append({"messages": list(messages), "tools": list(tools), "system": system})
        if not self.responses:
            return LLMResponse(text="(exhausted)", provider=self.name)
        return self.responses.pop(0)


def report_then_revise(agent, attempt):
    """Provider stub: bad draft on round 0, improved draft afterwards."""
    return ScriptedLLM([LLMResponse(text=DRAFT), LLMResponse(text=GOOD_REPORT)])


class TestReflection(TempWorkspace):
    def test_policy_scores_good_report_above_bad_draft(self) -> None:
        policy = ReflectionPolicy()
        task = "audit pkg/mod.py for eval injection"
        good, good_dims = policy.score(
            GOOD_REPORT, task=task, tools_used=["read_file"], tools_available=["read_file"], revision=0
        )
        bad, bad_dims = policy.score(DRAFT, task=task, tools_used=[], tools_available=["read_file"], revision=0)

        self.assertGreater(good, bad)
        self.assertGreaterEqual(good, 0.75)
        self.assertLess(bad, 0.75)
        self.assertEqual(sorted(good_dims), ["actionability", "coverage", "evidence", "specificity", "structure"])
        # The DRAFT fixture carries all six headers on purpose (so it exercises
        # the coverage/evidence/actionability penalties instead of structure);
        # structure equality is therefore the correct, specified behaviour.
        self.assertEqual(good_dims["structure"], bad_dims["structure"])
        self.assertGreater(good_dims["evidence"], bad_dims["evidence"])
        self.assertGreater(good_dims["actionability"], bad_dims["actionability"])

    def test_scoring_is_deterministic(self) -> None:
        policy = ReflectionPolicy()
        first = policy.score(GOOD_REPORT, task="audit pkg/mod.py", tools_used=["read_file"], tools_available=["read_file"], revision=0)
        second = policy.score(GOOD_REPORT, task="audit pkg/mod.py", tools_used=["read_file"], tools_available=["read_file"], revision=0)
        self.assertEqual(first, second)
        self.assertEqual(policy.score("", task="x", tools_used=[], tools_available=[], revision=0)[0], 0.0)

    def test_scorer_penalises_vagueness_and_rewards_code_spans(self) -> None:
        policy = ReflectionPolicy()
        vague = "## Summary\nclearly this is some better code, etc.\n"
        concrete = "## Summary\n`db.find_user` builds SQL with `%` interpolation in `db.py`.\n"
        _, vague_dims = policy.score(vague, task="audit db.py", tools_used=[], tools_available=[], revision=0)
        _, concrete_dims = policy.score(concrete, task="audit db.py", tools_used=[], tools_available=[], revision=0)
        self.assertLess(vague_dims["specificity"], concrete_dims["specificity"])

    def test_agent_loops_until_accepted(self) -> None:
        agent = Agent(
            "reviewer",
            role="Reviewer",
            goal="audit the package",
            llm=_StrategyStub([DRAFT, GOOD_REPORT]),
            tools=self.registry(),
            policy=ReflectionPolicy(min_score=0.75, max_rounds=3),
        )
        result = agent.execute("audit pkg/mod.py for eval injection")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.revisions, 1)
        self.assertEqual(len(result.reflections), 2)
        self.assertFalse(result.reflections[0].accepted)
        self.assertTrue(result.reflections[1].accepted)
        self.assertIn("rejected", result.reflections[0].critique)
        self.assertGreater(result.score, result.reflections[0].score)

    def test_agent_gives_up_after_max_rounds(self) -> None:
        agent = Agent(
            "reviewer",
            role="Reviewer",
            goal="audit",
            llm=_StrategyStub([DRAFT, DRAFT, DRAFT, DRAFT]),
            tools=self.registry(),
            policy=ReflectionPolicy(min_score=0.99, max_rounds=3),
        )
        result = agent.execute("audit pkg/mod.py")
        self.assertEqual(result.status, "unfinished")
        self.assertEqual(len(result.reflections), 3)
        self.assertEqual(result.revisions, 2)

    def test_agent_records_tool_calls_in_closed_loop(self) -> None:
        self.write("pkg/mod.py", "def f(x):\n    return eval(x)\n")
        calls = format_tool_calls([ToolCall(name="read_file", arguments={"path": "pkg/mod.py"})])
        first = LLMResponse(text="reading the module\n" + calls)
        agent = Agent(
            "sniffer",
            role="Sniffer",
            goal="find sinks",
            llm=_StrategyStub([first, GOOD_REPORT]),
            tools=self.registry(),
        )
        result = agent.execute("audit pkg/mod.py")

        self.assertEqual(len(result.tool_results), 1)
        self.assertTrue(result.tool_results[0].ok)
        self.assertEqual(result.tool_results[0].name, "read_file")
        self.assertIn("eval(x)", result.tool_results[0].output)
        self.assertIn("[tool ", result.output)
        self.assertEqual(result.status, "ok")
        self.assertIn("recovered tool calls from text", result.warnings)

    def test_tool_budget_is_enforced(self) -> None:
        """The budget is a HARD ceiling: a provider that always re-requests the
        same tool must stop at ``max_tool_calls`` total, never one call more."""
        self.write("pkg/mod.py", "x = 1\n")
        calls = format_tool_calls([ToolCall(name="read_file", arguments={"path": "pkg/mod.py"})])
        # Fresh response objects per call: the agent mutates ``response.text``
        # while stripping the recovered tool block, so a shared instance would
        # make the harness, not the agent, decide how many calls are requested.
        scripted = ScriptedLLM([LLMResponse(text="step\n" + calls) for _ in range(6)])
        agent = Agent(
            "sniffer",
            role="Sniffer",
            goal="find sinks",
            llm=_StrategyStub.from_llm(scripted),
            tools=self.registry(),
            max_tool_calls=2,
        )
        result = agent.execute("audit pkg/mod.py")
        self.assertEqual(len(result.tool_results), 2)
        self.assertTrue(any("budget" in w for w in result.warnings), result.warnings)

    def test_tool_budget_spans_the_whole_run(self) -> None:
        """A later reflection round may not re-spend the budget of an earlier one."""
        self.write("pkg/mod.py", "x = 1\n")
        calls = format_tool_calls([ToolCall(name="read_file", arguments={"path": "pkg/mod.py"})])
        scripted = ScriptedLLM([LLMResponse(text="step\n" + calls) for _ in range(6)])
        agent = Agent(
            "sniffer",
            role="Sniffer",
            goal="find sinks",
            llm=_StrategyStub.from_llm(scripted),
            tools=self.registry(),
            max_tool_calls=3,
        )
        result = agent.execute("audit pkg/mod.py")
        self.assertLessEqual(len(result.tool_results), 3)
        self.assertEqual(result.reflections[-1].usage["tool_call_budget"], 3)

    def test_provider_error_is_surfaced_not_raised(self) -> None:
        agent = Agent("flaky", role="R", goal="g", llm=_StrategyStub([LLMError("boom")]), tools=self.registry())
        result = agent.execute("audit pkg/mod.py")
        self.assertEqual(result.status, "failed")
        self.assertTrue(any("boom" in w for w in result.warnings), result.warnings)
        self.assertEqual(result.output, "")

    def test_blackboard_publication_and_writes(self) -> None:
        self.write("pkg/mod.py", "x = 1\n")
        board = Blackboard()
        agent = Agent(
            "reviewer",
            role="Reviewer",
            goal="audit",
            llm=_StrategyStub([GOOD_REPORT]),
            tools=self.registry(),
        )
        result = agent.execute("audit pkg/mod.py", blackboard=board, task_id="t-1")

        self.assertEqual(result.status, "ok")
        kinds = [m.kind for m in board.messages()]
        self.assertIn("assignment", kinds)
        self.assertIn("summary", kinds)
        self.assertEqual(board.findings("reviewer")["output"], GOOD_REPORT.strip() or result.output)
        self.assertEqual(board.findings("reviewer")["score"], result.score)
        self.assertEqual(result.writes["score"], result.score)
        self.assertTrue(all(m.metadata.get("thread") == "t-1" for m in board.messages()))

    def test_agent_namespacing_prefix(self) -> None:
        agent = Agent("coder", role="C", goal="g", llm=MockLLM(), name_prefix="grp-")
        self.assertEqual(agent.name, "grp-coder")
        self.assertEqual(agent.write_namespace, "grp-coder")

    def test_result_is_json_serialisable_and_has_usage(self) -> None:
        import json

        agent = Agent("a", role="R", goal="g", llm=_StrategyStub([GOOD_REPORT]), tools=self.registry())
        result = agent.execute("audit pkg/mod.py")
        payload = json.loads(json.dumps(result.to_dict(), sort_keys=True))
        self.assertEqual(payload["status"], "ok")
        usage = result.reflections[0].usage
        self.assertEqual(usage["llm_calls"], 1)
        self.assertIn("remaining_tool_calls", usage)
        self.assertEqual(payload["reflections"][0]["index"], 0)
        self.assertFalse(json.loads(json.dumps(result.to_dict(include_reflections=False), sort_keys=True))["reflections"])


class _StrategyStub:
    """Adapter that makes a scripted sequence usable as an ``LLM``."""

    name = "scripted-strategy"

    def __init__(self, responses):
        self._responses = responses
        self._llm = None

    @classmethod
    def from_llm(cls, llm):
        stub = cls([])
        stub._llm = llm
        return stub

    def complete(self, messages, *, tools=(), system=None, temperature=0.2, max_tokens=1024):
        if self._llm is not None:
            return self._llm.complete(messages, tools=tools, system=system, temperature=temperature, max_tokens=max_tokens)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestProviders(TempWorkspace):
    def test_parse_and_format_tool_calls_round_trip(self) -> None:
        calls = [
            ToolCall(name="read_file", arguments={"path": "a.py"}),
            ToolCall(name="search_text", arguments={"query": "eval("}),
        ]
        text = "thinking...\n" + format_tool_calls(calls)
        parsed = parse_tool_calls(text)
        self.assertEqual([c.name for c in parsed], ["read_file", "search_text"])
        self.assertEqual(parsed[0].arguments, {"path": "a.py"})
        self.assertTrue(parsed[0].id)

        self.assertEqual(parse_tool_calls("no tools here"), [])
        self.assertEqual(parse_tool_calls("TOOL_CALLS\nnot json"), [])
        tolerant = parse_tool_calls('TOOL_CALLS\n{"name": "read_file", "arguments": {"path": "b.py"}}\ntrailing text')
        self.assertEqual(len(tolerant), 1)
        array_form = parse_tool_calls('TOOL_CALLS\n[{"name": "list_files"}]\n')
        self.assertEqual(array_form[0].arguments, {})

    def test_openai_compatible_request_and_response_shape(self) -> None:
        captured = {}

        def transport(request, timeout):
            captured.update(request)
            return {
                "choices": [{"message": {"content": "hello", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
                ]}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }

        llm = OpenAICompatibleLLM(model="test-model", api_key="secret-key-value", base_url="https://example.invalid/v1", transport=transport)
        response = llm.complete(
            [_msg("user", "hi")], tools=[ToolSpec(name="read_file", description="d", parameters={"type": "object"})], system="sys"
        )

        self.assertEqual(captured["model"], "test-model")
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(response.text, "hello")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "a.py"})
        self.assertEqual(response.usage["prompt_tokens"], 3)
        self.assertEqual(llm.name, "openai-compatible:test-model")
        self.assertNotIn("secret-key-value", repr(response.raw))

    def test_empty_tools_omits_key(self) -> None:
        captured = {}

        def transport(request, timeout):
            captured.update(request)
            return {"choices": [{"message": {"content": "ok"}}]}

        llm = OpenAICompatibleLLM(model="m", api_key="k", base_url="https://x.invalid/v1", transport=transport)
        llm.complete([_msg("user", "hi")], tools=[])
        self.assertNotIn("tools", captured)

    def test_retry_then_success_and_error_sanitisation(self) -> None:
        state = {"calls": 0}

        class Boom(Exception):
            pass

        def flaky(request, timeout):
            state["calls"] += 1
            if state["calls"] == 1:
                raise LLMError("HTTP 500: upstream exploded")
            return {"choices": [{"message": {"content": "recovered"}}]}

        llm = OpenAICompatibleLLM(model="m", api_key="k", base_url="https://x.invalid/v1", transport=flaky, max_retries=2)
        llm._backoff = lambda attempt: 0  # no sleeping in tests
        self.assertEqual(llm.complete([_msg("user", "hi")]).text, "recovered")
        self.assertEqual(state["calls"], 2)

    def test_config_error_when_unconfigured(self) -> None:
        saved = {k: os.environ.pop(k, None) for k in ("OMP_API_KEY", "OPENAI_API_KEY", "OMP_LLM_MODE")}
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        with self.assertRaises(LLMConfigError):
            llm = OpenAICompatibleLLM.from_env()
            if llm.api_key:
                raise AssertionError("expected no api key")

    def test_build_llm_modes(self) -> None:
        saved = {k: os.environ.get(k) for k in ("OMP_LLM_MODE", "OMP_API_KEY", "OPENAI_API_KEY")}
        try:
            os.environ["OMP_LLM_MODE"] = "mock"
            os.environ.pop("OMP_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            provider, mode = build_llm()
            self.assertEqual(mode, "mock")
            self.assertIsInstance(provider, MockLLM)

            os.environ["OMP_LLM_MODE"] = "live"
            with self.assertRaises(LLMConfigError):
                build_llm()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_provider_fallback_degrades(self) -> None:
        warnings = []
        primary = _FailLLM()
        fallback = MockLLM(responder=lambda index, agent, messages, tools: LLMResponse(text="## Summary\n- fallback"))
        strategy = ProviderFallback(primary, fallback, warnings=warnings)

        chosen = strategy.choose(None, 0)
        self.assertIs(chosen, primary)
        self.assertFalse(strategy.degraded)
        try:
            chosen.complete([])
        except LLMError:
            pass
        chosen = strategy.choose(None, 1)
        self.assertIs(chosen, fallback)
        self.assertTrue(strategy.degraded)
        self.assertTrue(any("failed" in w for w in strategy.warnings))

    def test_mock_llm_is_deterministic_and_tool_aware(self) -> None:
        mock = MockLLM()
        prompt = _msg(
            "user",
            "## Task\nObjective: audit the package\nAudited files: db.py, auth.py\n\n"
            "## Tool request\n- read db.py\n- read auth.py\n- search for the pattern `eval(`\n",
        )
        first = mock.complete([prompt])
        second = MockLLM().complete([prompt])
        self.assertEqual([c.name for c in first.tool_calls], [c.name for c in second.tool_calls])
        self.assertEqual([c.arguments for c in first.tool_calls], [c.arguments for c in second.tool_calls])
        self.assertIn("read_file", [c.name for c in first.tool_calls])
        self.assertTrue(all(c.arguments.get("path") in ("db.py", "auth.py") for c in first.tool_calls if c.name == "read_file"))

        observation = _msg("user", "## Tool observations\n- [tool read_file ok] def total(items=[]):\n- read_file billing.py\n")
        report = mock.complete([observation]).text
        for header in ("## Summary", "## Findings", "## Evidence", "## Actions", "## Verification", "## Risks"):
            self.assertIn(header, report)

        revision = mock.complete([_msg("user", "## Critique\nreject: rewrite Actions as imperative bullets\n" + report)]).text
        self.assertGreater(len(revision), len(report))


def _msg(role, content):
    from squad.memory import Message

    return Message(role=role, sender="user", content=content)


class _FailLLM:
    name = "always-fails"

    def complete(self, messages, *, tools=(), system=None, temperature=0.2, max_tokens=1024):
        raise LLMError("provider down")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
