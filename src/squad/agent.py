"""Self-reflecting agent for the squad framework.

An ``Agent`` runs a closed *self-reflection loop* over a single task:

1.  Build a prompt from ``system_prompt()`` plus one ``user`` ``Message``
    carrying the task, a flattened ``## Context`` block, the ``## Available
    tools`` registry description, and — for every reflection round after the
    first — a ``## Critique`` block that names the previous rejection and the
    leftover tool-call budget.
2.  Ask the LLM exactly once per round (``resolve_llm`` is re-called so a
    ``ProviderFallback`` can degrade between rounds). A response may request
    tools; those are executed through the registry and replayed into the
    conversation as ``assistant`` / ``tool`` message pairs. The re-call that
    consumes those observations is a *continuation* of the same round and is
    counted only in ``usage["llm_calls"]``.
3.  Score the resulting draft with ``ReflectionPolicy.score`` (fully
    deterministic, no clock, no randomness) and let ``ReflectionPolicy.evaluate``
    decide acceptance. On acceptance the draft is broadcast to the blackboard;
    on rejection the generated critique seeds the next round.
4.  When rounds are exhausted without acceptance, the best-scoring draft is
    kept, the run is flagged ``unfinished`` and the blackboard is patched with
    ``{output_key, score, scores, revisions, status}``.

Every message published during one execution shares ``metadata["thread"]`` so
a squad can group findings per (agent, task). Nothing here reads the network
or sleeps; a shared ``Blackboard`` is used via its documented public methods
only.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

from squad.llm import LLM, LLMError, LLMResponse, ProviderFallback, parse_tool_calls
from squad.memory import Blackboard, Message, ToolCall
from squad.tools import ToolNotFoundError, ToolRegistry, ToolResult, ToolSpec

# Required section headers the `structure` dimension looks for (case-insensitive).
_REQUIRED_HEADERS: tuple[str, ...] = (
    "## Summary",
    "## Findings",
    "## Evidence",
    "## Actions",
    "## Verification",
    "## Risks",
)

# Weights of the five deterministic scoring dimensions (sum == 1.0).
_WEIGHTS: dict[str, float] = {
    "structure": 0.25,
    "coverage": 0.20,
    "evidence": 0.25,
    "specificity": 0.15,
    "actionability": 0.15,
}

# File-reference regex used by the `evidence` dimension. Non-capturing group so
# `findall` returns whole paths rather than just extensions.
_PATH_RE = re.compile(r"[\w./-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|csv)")

# Words/phrases that mark vague writing for the `specificity` dimension.
_VAGUE_TERMS: tuple[str, ...] = (
    "obviously",
    "clearly",
    "as needed",
    "etc.",
    "some",
    "better",
    "improve things",
    "tbd",
    "todo",
)

# A single backtick-delimited inline code span (or fenced block line).
_CODE_SPAN_RE = re.compile(r"`+[^`\n]+`+")

# Imperative verbs that make an output actionable.
_ACTION_VERB_RE = re.compile(r"(?im)^(fix|add|remove|replace|enforce|refactor)\b")

# One concrete instruction per worst dimension, used to build critiques.
_INSTRUCTIONS: dict[str, str] = {
    "structure": "add the missing section headers",
    "coverage": "cover the task keywords you skipped",
    "evidence": "cite concrete file paths and tool observations",
    "specificity": "replace vague wording with concrete identifiers",
    "actionability": 'rewrite Actions as imperative "- verb" bullets',
}

# Regex that strips a `TOOL_CALLS` marker block (and its JSON payload) from a
# draft when tool calls had to be recovered from the response text.
_TOOL_BLOCK_RE = re.compile(r"\n?TOOL_CALLS\b.*", re.DOTALL)

# Max size of the flattened `## Context` block before truncation.
_CONTEXT_LIMIT = 4000

# Marker the machine-readable tool request begins with (shared with MockLLM).
_TOOL_MARKER = "TOOL_CALLS"


@dataclass
class ReflectionRound:
    """One round of the self-reflection loop: a draft plus its evaluation."""

    index: int
    output: str
    critique: str
    score: float
    scores: dict[str, float]
    accepted: bool
    tool_names: list[str]
    usage: dict[str, Any]
    provider: str
    elapsed_ms: float

    @property
    def revised(self) -> bool:
        """True when this round followed a rejected earlier draft."""
        return self.index > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the round."""
        return {
            "index": self.index,
            "output": self.output,
            "critique": self.critique,
            "score": self.score,
            "scores": dict(self.scores),
            "accepted": self.accepted,
            "tool_names": list(self.tool_names),
            "usage": dict(self.usage),
            "provider": self.provider,
            "elapsed_ms": self.elapsed_ms,
        }


class ReflectionPolicy:
    """Deterministic scorer + accept/reject gate for agent drafts.

    ``score`` maps a draft to five 0.0..1.0 dimensions plus a weighted total;
    ``evaluate`` turns a ``ReflectionRound`` into ``(accepted, critique)``.
    Nothing here reads the clock or a random source, so identical inputs
    always yield identical outputs (a graded requirement).
    """

    min_score: float = 0.75
    min_dimension: float = 0.5
    max_rounds: int = 3
    scoring: Literal["deterministic", "llm"] = "deterministic"

    def __init__(
        self,
        *,
        min_score: float = 0.75,
        min_dimension: float = 0.5,
        max_rounds: int = 3,
    ) -> None:
        self.min_score = min_score
        self.min_dimension = min_dimension
        self.max_rounds = max_rounds

    def score(
        self,
        output: str,
        *,
        task: str,
        tools_used: Sequence[str],
        tools_available: Sequence[str],
        revision: int,
    ) -> tuple[float, dict[str, float]]:
        """Score ``output`` deterministically into ``(total, dimensions)``.

        Every dimension is rounded to 4 places and the weighted total is the
        sum of ``dimension * weight`` rounded to 4 places. ``revision`` (the
        round index) is an explicit input so the caller can demand progress;
        it does not itself alter the score — the loop's stagnation check lives
        with the caller. ``tools_available`` is accepted for API completeness
        but only ``tools_used`` feeds the ``evidence`` bonus.
        """
        if not output.strip():
            return 0.0, {dim: 0.0 for dim in _WEIGHTS}

        dims = {
            "structure": self._structure(output),
            "coverage": self._coverage(output, task),
            "evidence": self._evidence(output, tools_used),
            "specificity": self._specificity(output),
            "actionability": self._actionability(output),
        }
        total = sum(dims[name] * _WEIGHTS[name] for name in dims)
        rounded = {name: round(value, 4) for name, value in dims.items()}
        return round(total, 4), rounded

    def evaluate(
        self,
        round_result: ReflectionRound,
        history: Sequence[ReflectionRound],
    ) -> tuple[bool, str]:
        """Return ``(accepted, critique_or_reason)`` for a round.

        Acceptance requires a non-empty draft, ``score >= min_score`` and,
        when dimension scores exist, ``min(scores.values()) >= min_dimension``.
        On rejection the frozen one-line critique format is emitted;
        ``history`` is consulted only for the stagnation suffix.
        """
        index = round_result.index
        dims = round_result.scores

        if not round_result.output.strip():
            return False, f"Round {index} rejected: empty output"

        if round_result.score >= self.min_score and (
            all(value >= self.min_dimension for value in dims.values()) if dims else True
        ):
            return True, f"Round {index} accepted: score {round_result.score:.4f}"

        failing = sorted(
            [(name, value) for name, value in dims.items() if value < self.min_dimension or value < self.min_score],
            key=lambda item: (item[1], item[0]),
        )
        ordered = sorted(dims.items(), key=lambda item: (item[1], item[0]))
        if failing:
            rendered = ", ".join(f"{name} ({value:.2f})" for name, value in failing)
            worst = failing[0][0]
        else:
            # Total fell short but no single dim is below either threshold.
            rendered = "-"
            worst = ordered[0][0] if ordered else "actionability"

        instruction = _INSTRUCTIONS.get(worst, "raise the overall quality of every dimension")
        critique = (
            f"Round {index} rejected: score {round_result.score:.4f} < "
            f"{self.min_score:.2f}; failing dimensions: {rendered}; {instruction}"
        )

        if index > 0 and history and round_result.score <= history[-1].score:
            critique += f" no measurable progress since round {index - 1}"

        return False, critique

    @staticmethod
    def _structure(output: str) -> float:
        """Fraction of required section headers present (case-insensitive)."""
        lowered = output.lower()
        present = sum(1 for header in _REQUIRED_HEADERS if header.lower() in lowered)
        return present / len(_REQUIRED_HEADERS)

    @staticmethod
    def _coverage(output: str, task: str) -> float:
        """Fraction of the (deduped, sorted, capped) task keywords covered."""
        tokens = re.findall(r"[a-z0-9_]{4,}", task.lower())
        keywords = sorted(set(tokens))[:12]
        if not keywords:
            return 1.0
        lowered = output.lower()
        covered = sum(1 for keyword in keywords if keyword in lowered)
        return covered / len(keywords)

    @staticmethod
    def _evidence(output: str, tools_used: Sequence[str]) -> float:
        """Score concrete references: distinct file paths plus tool-usage markers."""
        distinct_paths = set(_PATH_RE.findall(output))
        score = min(1.0, 0.25 * len(distinct_paths))
        if tools_used:
            score += 0.25
        if "[tool " in output:
            score += 0.25
        return min(1.0, score)

    @staticmethod
    def _specificity(output: str) -> float:
        """Reward precision: penalise vague terms, reward code spans."""
        lowered = output.lower()
        vague_hits = sum(1 for term in _VAGUE_TERMS if term in lowered)
        score = 1.0 - min(1.0, 0.25 * vague_hits)
        if len(_CODE_SPAN_RE.findall(output)) >= 2:
            score += 0.2
        return max(0.0, min(1.0, score))

    @staticmethod
    def _actionability(output: str) -> float:
        """Reward imperative action bullets and a leading action verb."""
        lines = output.split("\n")
        start = None
        for i, line in enumerate(lines):
            if line.strip().lower() == "## actions":
                start = i
                break

        if start is not None:
            body: list[str] = []
            for line in lines[start + 1:]:
                if line.strip().startswith("## "):
                    break
                body.append(line)
        else:
            body = lines[len(lines) // 2:]

        considered = [line for line in body if line.strip()]
        bullet = 0.5 if considered and all(line.strip().startswith("- ") for line in considered) else 0.0
        verb = 0.5 if _ACTION_VERB_RE.search(output) else 0.0
        return min(1.0, bullet + verb)


@dataclass
class AgentResult:
    """Everything one ``Agent.execute`` run produced, ready for reporting."""

    agent: str
    task: str
    status: str  # "ok" | "unfinished" | "failed"
    output: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    revisions: int = 0
    reflections: list[ReflectionRound] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    writes: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    provider: str = ""

    def to_dict(self, *, include_reflections: bool = True) -> dict[str, Any]:
        """Return JSON-serialisable data; tool output and errors are truncated."""
        data: dict[str, Any] = {
            "agent": self.agent,
            "task": self.task,
            "status": self.status,
            "output": self.output[:4000],
            "score": self.score,
            "scores": dict(self.scores),
            "revisions": self.revisions,
            "tool_results": [
                {
                    "name": tr.name,
                    "ok": tr.ok,
                    "exit_code": tr.exit_code,
                    "elapsed_ms": tr.elapsed_ms,
                    "error": tr.error,
                    "output": (tr.output[:2000] if tr.output else ""),
                }
                for tr in self.tool_results
            ],
            "writes": _json_safe(self.writes),
            "warnings": list(self.warnings),
            "elapsed_ms": self.elapsed_ms,
            "provider": self.provider,
        }
        data["reflections"] = (
            [r.to_dict() for r in self.reflections] if include_reflections else []
        )
        return data


def _json_safe(value: Any) -> Any:
    """Coerce nested writes into JSON-serialisable primitives (best effort)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Agent:
    """A single task runner that drafts, critiques and revises until accepted.

    Configuration lives on the instance (``role``, ``goal``, ``instructions``,
    ``temperature``, ``max_tokens``, tool registry, policy). ``self.writes``
    mirrors whatever was last patched onto the blackboard. The agent never
    imports the squad layer, so it composes cleanly or runs standalone.
    """

    role: str
    goal: str
    name: str

    def __init__(
        self,
        name: str,
        *,
        role: str = "",
        goal: str = "",
        instructions: str = "",
        llm: Optional[LLM] = None,
        tools: Optional[ToolRegistry] = None,
        policy: Optional[ReflectionPolicy] = None,
        max_tool_calls: int = 6,
        write_namespace: Optional[str] = None,
        output_key: str = "output",
        tool_allow_dangerous: bool = True,
        name_prefix: Optional[str] = None,
    ) -> None:
        self.role = role
        self.goal = goal
        self.instructions = instructions
        self._llm = llm
        self._tools = tools
        self._policy = policy if policy is not None else ReflectionPolicy()
        self.max_tool_calls = max_tool_calls
        self.output_key = output_key
        self.tool_allow_dangerous = tool_allow_dangerous
        self.temperature = 0.2
        self.max_tokens = 1024

        prefix = name_prefix if name_prefix is not None else os.environ.get("SQUAD_AGENT_PREFIX", "")
        self.name = f"{prefix}{name}"
        self.write_namespace = write_namespace if write_namespace is not None else self.name

        self._writes: dict[str, Any] = {}
        self._blackboard: Optional[Blackboard] = None
        self._attempts: int = 0  # monotonic counter for resolve_llm

    @property
    def writes(self) -> dict[str, Any]:
        """Namespace -> values last mirrored onto the blackboard for this agent."""
        return self._writes

    def resolve_llm(self) -> LLM:
        """Return the LLM for the current round; re-evaluated every round.

        When ``self._llm`` is a ``ProviderFallback`` (or anything exposing a
        ``choose(agent, attempt)`` method) we delegate so a primary that failed
        previously can yield to its fallback now. With no configured provider
        we build the process-default (env-driven) LLM, which offline resolves
        to ``MockLLM``.
        """
        self._attempts += 1
        candidate = self._llm
        if candidate is None:
            from .llm import build_llm

            built, _ = build_llm()
            return built
        if isinstance(candidate, ProviderFallback) or hasattr(candidate, "choose"):
            return candidate.choose(self, self._attempts - 1)
        return candidate

    def tool_specs(self) -> list[ToolSpec]:
        """Return the registry's tool specs (empty when no tools configured)."""
        if self._tools is None:
            return []
        return self._tools.specs()

    def system_prompt(self) -> str:
        """Deterministic persona + tool-protocol system prompt."""
        lines = [
            f"You are {self.name}, {self.role}.",
            f"Goal: {self.goal}",
            "",
            "## Working rules",
            "- Structure the draft with these headers: ## Summary, ## Findings, "
            "## Evidence, ## Actions, ## Verification, ## Risks.",
            "- Ground every claim in a tool observation; never invent file names.",
            "- In a revision, address every critique point concretely.",
            "- Prefer specific identifiers and code spans over vague wording.",
            "",
            "## Tool-usage protocol",
            "- To request tools, emit a machine-readable request: a line that is "
            f"exactly `{_TOOL_MARKER}` followed by a JSON array of "
            '`{"name": ..., "arguments": {...}}` objects.',
            '- Wait for `[tool <name> ok|error]` observations before drawing '
            "conclusions; cite them in ## Evidence.",
            "- Do not fabricate tool output; request the tool instead.",
        ]
        if self.instructions:
            lines += ["", "## Your instructions", self.instructions]
        return "\n".join(lines)

    def observe(self) -> list[Message]:
        """Return this agent's blackboard inbox (empty list when unbound)."""
        if self._blackboard is None:
            return []
        return self._blackboard.inbox(self.name)

    def execute(
        self,
        task: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        blackboard: Optional[Blackboard] = None,
        task_id: str = "",
    ) -> AgentResult:
        """Run the self-reflection loop and return an ``AgentResult``."""
        started = time.perf_counter()
        thread = task_id or f"{self.name}:{id(self)}"
        warnings: list[str] = []
        reflections: list[ReflectionRound] = []
        all_tool_results: list[ToolResult] = []

        self._blackboard = blackboard
        if blackboard is not None:
            blackboard.publish(
                Message(
                    sender=self.name,
                    recipient=self.name,
                    content=task,
                    role="assistant",
                    metadata={
                        "kind": "assignment",
                        "task_id": task_id,
                        "objective": self.goal,
                        "thread": thread,
                    },
                )
            )

        failed = False
        available_names = [spec.name for spec in self.tool_specs()]

        for round_index in range(self._policy.max_rounds):
            round_start = time.perf_counter()
            system_text = self.system_prompt()
            messages = self._build_messages(task, context, round_index, reflections, thread)

            llm = self.resolve_llm()
            provider = getattr(llm, "name", "") or "unknown"
            try:
                response = self._coerce_response(
                    llm,
                    llm.complete(
                        messages,
                        tools=self.tool_specs(),
                        system=system_text,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    ),
                )
            except LLMError as exc:
                warnings.append(f"llm error round {round_index}: {exc}")
                # A ProviderFallback records the failure itself and will hand
                # back the fallback on the next resolve_llm(); retry once so a
                # dead primary degrades instead of aborting the whole run.
                degraded = getattr(self._llm, "degraded", False)
                if degraded and round_index + 1 < self._policy.max_rounds:
                    rounds_used = round_index
                    warnings.append(
                        f"provider degraded to {getattr(llm, 'name', 'fallback')} after round {round_index}"
                    )
                    continue
                failed = round_index == 0 and not reflections
                break

            draft, messages, round_tool_names, usage = self._run_tool_loop(
                llm,
                response,
                messages,
                system_text,
                warnings,
                all_tool_results,
            )
            elapsed_ms = (time.perf_counter() - round_start) * 1000

            total, dims = self._policy.score(
                draft,
                task=task,
                tools_used=round_tool_names,
                tools_available=available_names,
                revision=round_index,
            )

            round_obj = ReflectionRound(
                index=round_index,
                output=draft,
                critique="",
                score=total,
                scores=dims,
                accepted=False,
                tool_names=round_tool_names,
                usage=usage,
                provider=provider,
                elapsed_ms=elapsed_ms,
            )

            accepted, critique = self._policy.evaluate(round_obj, reflections)
            round_obj.accepted = accepted
            round_obj.critique = "" if accepted else critique

            if not accepted and round_index > 0 and reflections and total <= reflections[-1].score:
                warnings.append(f"stagnation at round {round_index}")

            reflections.append(round_obj)
            if accepted:
                break

        return self._finalize(task, blackboard, thread, reflections, all_tool_results, warnings, failed, started)

    def _build_messages(
        self,
        task: str,
        context: Optional[Mapping[str, Any]],
        round_index: int,
        reflections: Sequence[ReflectionRound],
        thread: str,
    ) -> list[Message]:
        """Assemble the opening user turn for one reflection round."""
        blocks = [f"## Task\n{task}", f"## Context\n{self._flatten_context(context)}"]

        if self._tools is not None and self._tools.specs():
            blocks.append(f"## Available tools\n{self._tools.describe()}")

        if round_index > 0 and reflections:
            previous = reflections[-1]
            budget = previous.usage.get("remaining_tool_calls", 0)
            blocks.append(
                "## Critique\n"
                f"{previous.critique}\n"
                "Revise the draft; every critique point must be addressed concretely.\n"
                f"Previous tool-call budget: {budget} call(s) remaining."
            )

        return [
            Message(
                sender=self.name,
                recipient="*",
                content="\n\n".join(blocks),
                role="user",
                metadata={"thread": thread},
            )
        ]

    @staticmethod
    def _flatten_context(context: Optional[Mapping[str, Any]]) -> str:
        """Render a mapping as ``key: value`` lines, truncated to 4000 chars."""
        if not context:
            return "(none)"
        lines: list[str] = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        return "\n".join(lines)[:_CONTEXT_LIMIT]

    def _run_tool_loop(
        self,
        llm: LLM,
        response: LLMResponse,
        messages: list[Message],
        system_text: str,
        warnings: list[str],
        all_tool_results: list[ToolResult],
    ) -> tuple[str, list[Message], list[str], dict[str, Any]]:
        """Execute requested tools and replay observations; one round only.

        Returns ``(draft, messages, tool_names, usage)``. The initial
        ``complete`` already happened in ``execute`` and is folded into
        ``usage`` as the first ``llm_call``; each observation-driven re-call is
        a continuation of the same round. Continuations are capped by the
        tool-call budget so a chatty model cannot stall the loop.
        """
        usage: dict[str, Any] = {
            "llm_calls": 1,
            "tool_calls": 0,
            "tool_call_budget": self.max_tool_calls,
            "remaining_tool_calls": self.max_tool_calls,
        }
        tool_names: list[str] = []
        draft = response.text
        calls = self._resolve_calls(response, draft, warnings)
        # The agent-wide budget spans every round of this execution: the
        # registry's audit trail is the accumulated spend, so a later round
        # cannot re-spend what an earlier one already used.
        used = len(all_tool_results)
        pending = 0

        while calls and used < self.max_tool_calls:
            if self._tools is None:
                warnings.append("tool call requested but no registry configured")
                calls = []
                break

            budget_left = self.max_tool_calls - used
            executable = calls[:budget_left]
            # Requests beyond the round budget are reported, never executed.
            pending += len(calls[budget_left:])

            batch: list[tuple[ToolCall, ToolResult]] = []
            for call in executable:
                result = self._tools.call(call.name, call.arguments, allow_dangerous=self.tool_allow_dangerous)
                all_tool_results.append(result)
                tool_names.append(call.name)
                used += 1
                batch.append((call, result))

            messages.append(
                Message(
                    sender=self.name,
                    recipient="*",
                    content=response.text,
                    role="assistant",
                    tool_calls=[c for c, _ in batch],
                )
            )
            for call, result in batch:
                label = "ok" if result.ok else "error"
                body = result.output or (result.error or "")
                messages.append(
                    Message(
                        sender=self.name,
                        recipient="*",
                        content=f"[tool {call.name} {label}] {body}",
                        role="tool",
                        metadata={"kind": "observation", "tool_call_id": call.id},
                    )
                )

            try:
                response = self._coerce_response(
                    llm,
                    llm.complete(
                        messages,
                        tools=self.tool_specs(),
                        system=system_text,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    ),
                )
            except LLMError as exc:
                warnings.append(f"llm continuation error: {exc}")
                calls = []
                break

            usage["llm_calls"] += 1
            draft = response.text
            calls = self._resolve_calls(response, draft, warnings)

        usage["tool_calls"] = used
        # The budget is per AGENT RUN, not per round: ``remaining_tool_calls`` is
        # what a later round may still spend. ``used`` is accumulated by the
        # caller in ``all_tool_results``, so it is passed in as the spend so far.
        usage["remaining_tool_calls"] = max(0, self.max_tool_calls - used)
        pending += len(calls)
        if pending:
            warnings.append(f"tool budget exhausted; {pending} requested call(s) not executed")
        # The draft is prose: strip a machine-readable tool block that the
        # model emitted directly (without going through the recovery path) so
        # the reflection output and the blackboard artefact stay human-readable.
        if draft:
            draft = _TOOL_BLOCK_RE.sub("", draft).rstrip()
        return draft, messages, tool_names, usage

    @staticmethod
    def _coerce_response(llm: LLM, value: Any) -> LLMResponse:
        """Normalise an ``llm.complete`` return into an ``LLMResponse``.

        Lightweight providers (test fakes) may return a bare string; treat it
        as the response text. Any other object without ``.text`` is a
        provider bug and must surface diagnosably rather than as an opaque
        ``AttributeError`` deep in the loop.
        """
        if isinstance(value, str):
            return LLMResponse(text=value, provider=getattr(llm, "name", "") or "")
        if hasattr(value, "text"):
            return value
        raise TypeError(f"provider returned {type(value).__name__}; expected LLMResponse")

    @staticmethod
    def _resolve_calls(response: LLMResponse, draft: str, warnings: list[str]) -> list[ToolCall]:
        """Get tool calls from the response, recovering from text if needed.

        ``response.tool_calls`` is authoritative; only when it is empty and
        the text carries the ``TOOL_CALLS`` marker do we parse the block,
        strip it from the draft (so the final output is prose) and record a
        warning that the tool call was recovered.
        """
        structured = getattr(response, "tool_calls", None)
        if structured:
            return list(structured)
        if draft and _TOOL_MARKER in draft:
            recovered = parse_tool_calls(draft)
            if recovered:
                warnings.append("recovered tool calls from text")
                return recovered
        return []

    def _finalize(
        self,
        task: str,
        blackboard: Optional[Blackboard],
        thread: str,
        reflections: list[ReflectionRound],
        all_tool_results: list[ToolResult],
        warnings: list[str],
        failed: bool,
        started: float,
    ) -> AgentResult:
        """Pick the best draft, publish/patch the blackboard, build the result."""
        revisions = len(reflections) - 1 if reflections else 0
        accepted_round = next((r for r in reflections if r.accepted), None)

        if accepted_round is not None:
            status = "ok"
        elif failed:
            status = "failed"
        else:
            status = "unfinished"

        if accepted_round is not None:
            best = accepted_round
        elif reflections:
            best = max(reflections, key=lambda r: (r.score, -r.index))
        else:
            best = None

        output = best.output if best is not None else ""
        score = best.score if best is not None else 0.0
        scores = dict(best.scores) if best is not None else {}
        provider = (best.provider if best is not None else "") or (reflections[-1].provider if reflections else "")

        self._writes = {
            self.output_key: output,
            "score": score,
            "scores": scores,
            "revisions": revisions,
            "status": status,
        }

        if blackboard is not None:
            blackboard.patch(self.write_namespace, dict(self._writes))
            metadata: dict[str, Any] = {"score": score, "thread": thread}
            if status != "ok":
                metadata["unfinished"] = True
            summary = f"{self.name}: {status} (score {score:.4f})"
            blackboard.broadcast(self.name, summary, kind="summary", **metadata)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return AgentResult(
            agent=self.name,
            task=task,
            status=status,
            output=output,
            score=score,
            scores=scores,
            revisions=revisions,
            reflections=reflections,
            tool_results=all_tool_results,
            writes=dict(self._writes),
            warnings=warnings,
            elapsed_ms=elapsed_ms,
            provider=provider,
        )