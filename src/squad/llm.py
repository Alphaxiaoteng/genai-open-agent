"""Provider abstraction for squad: live OpenAI-compatible + deterministic mock.

``OpenAICompatibleLLM`` performs real HTTP with ``urllib.request`` only (never
``requests``): POST ``{base_url}/chat/completions`` with a payload carrying
``model``/``messages``/``max_tokens``/``temperature`` and ``tools`` only when
non-empty; transient failures (HTTP 429/500/502/503/504, ``URLError``,
``TimeoutError``) and transport-raised ``LLMError`` retry up to ``max_retries``
total attempts with ``self._backoff(attempt)`` sleep, then raise ``LLMError``.
The API key travels only in the ``Authorization`` header and is redacted from
any outgoing message. ``MockLLM`` is the offline provider: zero network, zero
credentials, zero randomness, zero timestamps.

Wire protocol (single source of truth, shared with the squad layer)
------------------------------------------------------------------
An agent prompt may request tools with the marker line ``TOOL_CALLS`` followed
by JSON payload: one or more ``{...}`` objects or ``[...]`` arrays of objects
``{"name": ..., "arguments": {...}}``. Non-JSON text after the marker is
allowed. ``parse_tool_calls`` scans from the LAST marker, decodes every
balanced span (string/escape aware), skips malformed fragments, preserves
order, and assigns ``id = "call_<n>"`` (1-based global) unless the payload
carries a non-empty string id, so ``parse_tool_calls(format_tool_calls(x)) ==
x`` round-trips. A call needs a non-empty ``name``; ``arguments`` defaults
``{}`` when absent or non-dict.

MockLLM default-responder priority (deterministic; documented contract)
----------------------------------------------------------------------
1. Report: the prompt has an observation block (``## Tool observations`` /
   ``## Observations`` header, or a ``[tool `` line) AND no ``TOOL_CALLS``
   marker AND no ``## Tool request`` header AND no ``## Critique`` -> six
   section text with headers ``## Summary``, ``## Findings``, ``## Evidence``,
   ``## Actions``, ``## Verification``, ``## Risks`` (all six always present),
   every bullet copied from observation lines, ``- none`` when a section has no
   source, ending ``Status: draft`` on round 0. ``OMP_MOCK_SCORE_STALL=1``
   (read inside the responder, never at import) makes the 2nd+ report
   byte-identical to the first so tests can prove stagnation detection.
2. Revision: the prompt has ``## Critique`` or a line starting with
   ``critique:`` -> rewrite the previous assistant draft: keep its section
   structure, append one ``- addressed: <bullet>`` per critique bullet under
   ``## Actions``, ensure at least one new concrete identifier from the
   critique appears under ``## Evidence``, guarantee the result is strictly
   longer than the draft (pad ``## Risks`` if needed), end with a
   ``Revised for: <summary>`` line then ``Status: revised``.
3. Tool request: a ``TOOL_CALLS`` marker OR a ``## Tool request`` header ->
   derive calls from the prompt. Explicit JSON tool objects after a marker are
   decoded by ``parse_tool_calls`` first; otherwise every path token matching
   ``[A-Za-z0-9_./-]+\\.(py|js|ts|json|md|txt|yaml|yml|toml)`` (first
   appearance, deduped) plus keyword intents drive derivation: ``read`` ->
   ``read_file`` per path; ``list files``/``inventory`` -> one ``list_files``;
   ``search``/``pattern`` -> ``search_text`` with the first backticked token
   after "pattern" (default ``eval(``); ``write``/``create``/``generate tests``
   -> ``write_file`` of ``tests/test_<stem>.py``; ``run``/``execute``/``verify``
   -> ``python_exec``. Never invents paths; no path AND no intent -> plan text.
4. Otherwise: a short deterministic plan skeleton derived from the prompt
   (``## Summary`` from the task line, ``## Actions`` = ``- read <path>`` per
   discovered path or ``- inspect the workspace``, ending ``Status: draft``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence

from .memory import Message, ToolCall

if TYPE_CHECKING:
    from .tools import ToolSpec  # string annotations only; avoids import cycles

__all__ = [
    "LLMResponse", "LLMError", "LLMConfigError", "LLM", "LLMStrategy",
    "OpenAICompatibleLLM", "MockLLM", "ProviderFallback", "build_llm",
    "parse_tool_calls", "format_tool_calls",
]

TOOL_CALL_MARKER = "TOOL_CALLS"
_REPORT_HEADERS = ("## Summary", "## Findings", "## Evidence", "## Actions",
                   "## Verification", "## Risks")
_OBS_HEADERS = ("## tool observations", "## observations")
_TOOL_REQUEST_HEADER = "## tool request"
_PATH_RE = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml)")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_BULLET_RE = re.compile(r"^\s*[-*]\s*(.+)$")
_CRITIQUE_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$")
_CRITIQUE_LINE_RE = re.compile(r"(?im)^\s*critique:")
_TOOL_NAME_RE = re.compile(r"\b(read_file|write_file|list_files|search_text|python_exec)\b")
_SRC_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"
_EXEC_CODE = "import pathlib;print(len(list(pathlib.Path('.').rglob('*.py'))))"
_WRITE_CONTENT = "# generated by agent-squad\n"
_RETRY_STATUSES = frozenset((429, 500, 502, 503, 504))
_REQUEST_META_KEYS = ("url", "headers")
_RISK_PAD = "- none beyond the addressed points"
_NON_CONCRETE = ("critique", "revision", "draft", "address", "revised")


class LLMError(RuntimeError):
    """Base error for all provider failures."""


class LLMConfigError(LLMError):
    """Missing or invalid provider configuration (e.g. no API key)."""


@dataclass
class LLMResponse:
    """Normalized provider reply: final text, requested tool calls, metadata."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    provider: str = ""


class LLM(Protocol):
    """Minimal provider interface consumed by the agent loop."""

    name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: "Sequence[ToolSpec]" = (),
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send the conversation and return one normalized response."""


class LLMStrategy(Protocol):
    """Chooses the provider per agent/attempt (e.g. fallback chains)."""

    def choose(self, agent: Any, attempt: int) -> Any:
        """Return the LLM for ``agent``'s ``attempt`` (None = no provider)."""


def _scan_balanced(text: str, start: int, opener: str) -> int:
    """Exclusive end index of the balanced span at ``start``; -1 on failure."""
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for index in range(start, len(text)):
        ch = text[index]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _call_from_dict(data: Any, index: int) -> Optional[ToolCall]:
    """Build one ToolCall (name required non-empty str; arguments default {})."""
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    raw_id = data.get("id")
    call_id = raw_id if isinstance(raw_id, str) and raw_id else f"call_{index}"
    return ToolCall(name=name, arguments=arguments, id=call_id)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse the wire protocol out of arbitrary text; never raises.

    Uses the LAST ``TOOL_CALLS`` marker, decodes every balanced ``{...}`` and
    ``[...]`` span after it (arrays expand their dict elements), skips malformed
    fragments, preserves order, assigns ``id = "call_<n>"`` (1-based global)
    unless the payload sets a non-empty string id, and returns ``[]`` when the
    marker is absent.
    """
    if not text:
        return []
    start = text.rfind(TOOL_CALL_MARKER)
    if start == -1:
        return []
    region = text[start + len(TOOL_CALL_MARKER):]
    calls: list[ToolCall] = []
    cursor = 0
    while cursor < len(region):
        brace = region.find("{", cursor)
        bracket = region.find("[", cursor)
        if brace == -1 and bracket == -1:
            break
        if brace == -1 or (0 <= bracket < brace):
            pos, opener = bracket, "["
        else:
            pos, opener = brace, "{"
        end = _scan_balanced(region, pos, opener)
        if end == -1:
            cursor = pos + 1
            continue
        try:
            decoded = json.loads(region[pos:end])
        except ValueError:
            cursor = pos + 1
            continue
        entries = decoded if opener == "[" and isinstance(decoded, list) else [decoded]
        for entry in entries:
            call = _call_from_dict(entry, len(calls) + 1)
            if call is not None:
                calls.append(call)
        cursor = end
    return calls


def format_tool_calls(calls: Sequence[ToolCall]) -> str:
    """Serialize calls into the wire block (marker + one JSON-array line).

    ``parse_tool_calls(format_tool_calls(x)) == x`` for parsed/mock calls;
    existing non-empty ids are preserved, empty ones become ``call_<n>``.
    """
    payload = [{"id": call.id or f"call_{index + 1}", "name": call.name,
                "arguments": call.arguments} for index, call in enumerate(calls)]
    return f"{TOOL_CALL_MARKER}\n{json.dumps(payload, ensure_ascii=False)}"


def _spec_field(spec: Any, attr: str) -> Any:
    """Read a ToolSpec field duck-typed (dataclass attr or mapping key)."""
    return spec.get(attr) if isinstance(spec, dict) else getattr(spec, attr, None)


def _encode_message(message: Message, auto_index: int) -> dict:
    """Map one Message to the OpenAI wire shape (role/tool_calls/tool_call_id)."""
    role = message.role or "assistant"
    if role == "tool":
        meta = message.metadata or {}
        call_id = str(meta.get("tool_call_id", "") or "")
        if not call_id and message.tool_calls:
            call_id = message.tool_calls[0].id
        return {"role": "tool", "tool_call_id": call_id, "content": message.content}
    if role == "assistant" and message.tool_calls:
        encoded = [{"id": call.id or f"call_{auto_index}_{inner}", "type": "function",
                    "function": {"name": call.name,
                                 "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                   for inner, call in enumerate(message.tool_calls)]
        return {"role": "assistant", "content": message.content, "tool_calls": encoded}
    return {"role": role, "content": message.content}


def _encode_messages(messages: Sequence[Message], system: Optional[str]) -> list[dict]:
    """Prepend a system prompt (when given) and encode every message."""
    result: list[dict] = []
    if system:
        result.append({"role": "system", "content": system})
    result.extend(_encode_message(message, index)
                  for index, message in enumerate(messages))
    return result


def _encode_tools(tools: "Sequence[ToolSpec]") -> list[dict]:
    """Render ToolSpecs as the OpenAI ``tools`` array (empty list when none)."""
    rendered: list[dict] = []
    for spec in tools or ():
        parameters = _spec_field(spec, "parameters")
        rendered.append({"type": "function", "function": {
            "name": _spec_field(spec, "name"),
            "description": _spec_field(spec, "description") or "",
            "parameters": parameters if isinstance(parameters, dict) else {}}})
    return rendered


def _parse_completion(data: dict, *, fallback_model: str, provider: str) -> LLMResponse:
    """Decode a chat-completions body into ``LLMResponse`` (tolerant of None)."""
    choices = data.get("choices") or [{}]
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    calls: list[ToolCall] = []
    for entry in (message.get("tool_calls") or []):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (ValueError, TypeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        raw_id = entry.get("id")
        call_id = raw_id if isinstance(raw_id, str) and raw_id else f"call_{len(calls) + 1}"
        calls.append(ToolCall(name=str(function.get("name") or ""), arguments=arguments, id=call_id))
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return LLMResponse(text=text, tool_calls=calls, raw=data,
                       model=str(data.get("model") or fallback_model),
                       usage=usage, provider=provider)


def _retryable(exc: BaseException) -> bool:
    """Transient HTTP status (429/5xx) or a connection/timeout failure."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRY_STATUSES
    return isinstance(exc, (urllib.error.URLError, TimeoutError))


def _describe(exc: Optional[BaseException]) -> str:
    """One-line error description; HTTPError bodies are read + truncated to 400."""
    if exc is None:
        return "provider request failed"
    if isinstance(exc, LLMError):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return f"HTTP {exc.code}: {body[:400]}" if body else f"HTTP {exc.code}: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def _default_transport(request: dict, timeout: float) -> dict:
    """urllib transport: POST the API subset of ``request``, return JSON body."""
    body = json.dumps({k: v for k, v in request.items()
                       if k not in _REQUEST_META_KEYS}).encode("utf-8")
    req = urllib.request.Request(request["url"], data=body,
                                 headers=dict(request["headers"]), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        decoded = json.loads(resp.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise LLMError("provider returned a non-object payload")
    return decoded


class OpenAICompatibleLLM:
    """OpenAI-compatible chat-completions provider over stdlib HTTP.

    ``transport`` is the injection point: ``transport(request: dict, timeout:
    float) -> dict`` returning the decoded JSON payload. ``request`` is the
    chat payload itself (top-level ``model``/``messages``/``max_tokens``/
    ``temperature`` and optional ``tools``) plus ``url``/``headers`` meta keys,
    so tests read ``request["model"]`` directly and never touch the network.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: Optional[Callable[[dict, float], dict]] = None,
        provider: str = "openai-compatible",
    ) -> None:
        """Configure the client; base_url gets its trailing slashes stripped."""
        self._model = model
        self._api_key = api_key or ""
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = float(timeout)
        self._max_retries = max(1, int(max_retries))
        self._transport = transport or _default_transport
        self._provider = provider

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenAICompatibleLLM":
        """Build from OMP_BASE_URL / OMP_API_KEY / OMP_MODEL / OMP_TIMEOUT.

        ``OMP_API_KEY`` falls back to ``OPENAI_API_KEY``; a missing/blank key
        raises ``LLMConfigError``. Explicit overrides win over the environment.
        """
        model = overrides.get("model") or os.environ.get("OMP_MODEL") or _DEFAULT_MODEL
        base_url = overrides.get("base_url") or os.environ.get("OMP_BASE_URL") or _DEFAULT_BASE_URL
        api_key = overrides.get("api_key")
        if api_key is None:
            api_key = os.environ.get("OMP_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not str(api_key).strip():
            raise LLMConfigError("no API key: set OMP_API_KEY (or OPENAI_API_KEY)")
        timeout = overrides.get("timeout")
        if timeout is None:
            raw = os.environ.get("OMP_TIMEOUT", "60")
            try:
                timeout = float(raw)
            except ValueError as exc:
                raise LLMConfigError(f"invalid OMP_TIMEOUT: {raw!r}") from exc
        kwargs: dict[str, Any] = {"model": model, "api_key": str(api_key).strip(),
                                  "base_url": base_url.rstrip("/"), "timeout": timeout}
        for key in ("max_retries", "transport", "provider"):
            if key in overrides:
                kwargs[key] = overrides[key]
        return cls(**kwargs)

    @property
    def name(self) -> str:
        """Stable identifier: ``"openai-compatible:" + model``."""
        return f"openai-compatible:{self._model}"

    def _backoff(self, attempt: int) -> float:
        """Seconds to sleep before the ``attempt``-th retry (test-injectable)."""
        return 0.5 * (2 ** attempt)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: "Sequence[ToolSpec]" = (),
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """POST /chat/completions with retry/backoff; parse into ``LLMResponse``."""
        payload: dict[str, Any] = {"model": self._model,
                                   "messages": _encode_messages(messages, system),
                                   "max_tokens": max_tokens, "temperature": temperature}
        tool_payload = _encode_tools(tools)
        if tool_payload:
            payload["tools"] = tool_payload  # never send "tools": []
        request = dict(payload)  # request IS the payload: model/messages/... at top level
        request["url"] = f"{self._base_url}/chat/completions"
        request["headers"] = {"Content-Type": "application/json",
                              "Authorization": f"Bearer {self._api_key}"}
        last: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                data = self._transport(request, self._timeout)
                if not isinstance(data, dict):
                    raise LLMError("provider returned a non-object payload")
                return _parse_completion(data, fallback_model=self._model, provider=self._provider)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last = exc
                if not _retryable(exc) or attempt + 1 >= self._max_retries:
                    break
                time.sleep(self._backoff(attempt))
            except LLMError as exc:  # injected transport failures retry too
                last = exc
                if attempt + 1 >= self._max_retries:
                    break
                time.sleep(self._backoff(attempt))
            except Exception as exc:  # unexpected transport error: no retry
                last = exc
                break
        raise LLMError(self._sanitize(_describe(last))) from last

    def _sanitize(self, message: str) -> str:
        """Belt-and-braces API key redaction (messages never carry it by design)."""
        if self._api_key and self._api_key in message:
            message = message.replace(self._api_key, "[redacted]")
        return message


def _last_prompt(messages: Sequence[Message]) -> str:
    """Content of the LAST user message (the agent's prompt)."""
    for message in reversed(list(messages)):
        if getattr(message, "role", "") == "user":
            return str(getattr(message, "content", "") or "")
    if messages:
        return str(getattr(messages[-1], "content", "") or "")
    return ""


def _has_observation(prompt: str) -> bool:
    """Whether the prompt carries an observation block."""
    lowered = prompt.lower()
    return (any(header in lowered for header in _OBS_HEADERS)
            or any(line.strip().startswith("[tool ") for line in prompt.splitlines()))


def _conversation_text(messages: Sequence[Message]) -> str:
    """Join the whole conversation for observation/critique detection.

    The agent replays tool results as ``role="tool"`` messages and may keep the
    original user prompt as the last ``user`` message, so scanning only
    ``_last_prompt`` would miss every observation. All roles are scanned, with
    the tool messages first (most direct evidence) and the user prompt last.
    """
    chunks: list[str] = []
    for message in messages:
        role = getattr(message, "role", "") or ""
        content = str(getattr(message, "content", "") or "")
        if not content:
            continue
        if role == "tool":
            if not content.lstrip().startswith("[tool "):
                content = f"[tool observation] {content}"
        chunks.append(content if role != "system" else content)
    return "\n".join(chunks)


def _observations_from_messages(messages: Sequence[Message]) -> str:
    """The observation block harvested from every ``tool`` message."""
    lines: list[str] = []
    for message in messages:
        if (getattr(message, "role", "") or "") != "tool":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        if not content.startswith("[tool "):
            content = f"[tool observation] {content}"
        lines.append(content)
    return "\n".join(lines)


def _observation_block(prompt: str) -> str:
    """Text of the observation block: header to next ``## ``, or the [tool lines."""
    lowered = prompt.lower()
    start = -1
    for header in _OBS_HEADERS:
        found = lowered.find(header)
        if found != -1 and (start == -1 or found < start):
            start = found
    if start != -1:
        end = lowered.find("\n## ", start)
        return prompt[start:end] if end != -1 else prompt[start:]
    return "\n".join(line for line in prompt.splitlines() if line.strip().startswith("[tool "))


def _has_critique(prompt: str) -> bool:
    """Whether the prompt carries a critique block (``## Critique`` / ``critique:``)."""
    if "## critique" in prompt.lower():
        return True
    return any(line.strip().lower().startswith("critique:") for line in prompt.splitlines())


def _critique_block(prompt: str) -> str:
    """Character-accurate slice of the critique block (never a line-number mixup)."""
    pos = prompt.lower().find("## critique")
    if pos == -1:
        match = _CRITIQUE_LINE_RE.search(prompt)
        pos = match.start() if match else -1
    return prompt[pos:] if pos != -1 else ""


def _is_round_zero(messages: Sequence[Message]) -> bool:
    """Round 0 heuristic: no earlier report-shaped assistant message."""
    return not [m for m in list(messages)[:-1]
                if getattr(m, "role", "") == "assistant"
                and "## Summary" in str(getattr(m, "content", "") or "")]


def _bullet(line: str) -> str:
    """Strip a leading bullet marker; return the remainder trimmed."""
    match = _BULLET_RE.match(line)
    return match.group(1).strip() if match else line.strip()


def _unique(items: list[str]) -> list[str]:
    """Deduplicate preserving first-appearance order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _report_from_observations(prompt: str, round_zero: bool) -> str:
    """Six-section report; every bullet drawn strictly from observation lines."""
    block = _observation_block(prompt)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    tools = _unique([name for line in lines for name in _TOOL_NAME_RE.findall(line)])
    filenames = _unique([m.group(0) for m in _PATH_RE.finditer(block)])
    source = [_bullet(ln) for ln in lines
              if _PATH_RE.search(ln) or "exit_code" in ln or "line" in ln.lower()]
    summary = (source[0] if source else
               ("reviewed: " + filenames[0] if filenames else "tool observations collected"))
    sections = {
        "## Summary": [summary[:200]],
        "## Findings": [f"observed activity from tools: {', '.join(tools)}"] if tools else [],
        "## Evidence": source[:5],
        "## Actions": [f"re-run {name}" for name in tools[:4]],
        "## Verification": [f"re-run {', '.join(tools)} over the same targets"] if tools else [],
        "## Risks": [],
    }
    out: list[str] = []
    for header in _REPORT_HEADERS:
        out.append(header)
        items = sections[header]
        if items:
            out.extend(f"- {str(item)[:200]}" for item in items)
        else:
            out.append("- none")
    if round_zero:
        out.append("Status: draft")
    return "\n".join(out)


def _plan_text(prompt: str) -> str:
    """Deterministic plan skeleton (used when no observation/critique/tool fits)."""
    first_line = next((ln.strip() for ln in prompt.splitlines() if ln.strip()), "")
    title = _bullet(re.sub(r"^##\s*Task:?\s*", "", first_line)) or "plan the requested work"
    paths = _unique([m.group(0) for m in _PATH_RE.finditer(prompt)])
    tools = _unique(_TOOL_NAME_RE.findall(prompt))
    actions = [f"- read {path}" for path in paths[:3]] or ["- inspect the workspace"]
    verification = f"- re-run {tools[0]}" if tools else "- re-run the tools after edits"
    return "\n".join(["## Summary", f"- plan: {title}"[:200],
                      "## Findings", "- none yet",
                      "## Evidence", "- none",
                      "## Actions", *actions,
                      "## Verification", verification,
                      "## Risks", "- draft only, no tool observations yet",
                      "Status: draft"])


def _derive_tool_calls(prompt: str) -> list[ToolCall]:
    """Mock derivation from path tokens + keyword intents; never invents paths."""
    paths = _unique([m.group(0) for m in _PATH_RE.finditer(prompt)])
    lowered = prompt.lower()
    intents: list[str] = []
    if re.search(r"\bread\b", lowered) and paths:
        intents.append("read")
    if "list files" in lowered or "inventory" in lowered:
        intents.append("list")
    if "search" in lowered or "pattern" in lowered:
        intents.append("search")
    if "write" in lowered or "create" in lowered or "generate tests" in lowered:
        intents.append("write")
    if re.search(r"\brun\b", lowered) or "execute" in lowered or "verify" in lowered:
        intents.append("run")
    if not paths and not intents:
        return []
    calls: list[ToolCall] = []

    def _add(name: str, arguments: dict) -> None:
        calls.append(ToolCall(name=name, arguments=arguments, id=f"call_{len(calls) + 1}"))

    if "read" in intents:
        for path in paths:
            _add("read_file", {"path": path})
    if "list" in intents:
        _add("list_files", {})
    if "search" in intents:
        pat_idx = lowered.find("pattern")
        after = prompt[pat_idx:] if pat_idx != -1 else prompt
        ticks = _BACKTICK_RE.findall(after)
        _add("search_text", {"query": ticks[0] if ticks else "eval("})
    if "write" in intents and paths:
        stem = paths[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        _add("write_file", {"path": f"tests/test_{stem}.py", "content": _WRITE_CONTENT})
    if "run" in intents:
        _add("python_exec", {"code": _EXEC_CODE})
    return calls


def _parse_sections(draft: str) -> dict[str, list[str]]:
    """Split a report/plan draft into header -> bullet lines (insertion order)."""
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in draft.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Status:"):
            continue
        if stripped.startswith("## "):
            current = stripped
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(stripped)
    return sections


def _critique_bullets(critique: str) -> list[str]:
    """Bullet items of the critique block (numbered/bulleted, else plain lines)."""
    bullets = [m.group(1).strip() for m in _CRITIQUE_BULLET_RE.finditer(critique)
               if m.group(1).strip() and not m.group(1).strip().startswith("##")]
    if bullets:
        return bullets
    return [ln.strip() for ln in critique.splitlines() if ln.strip()
            and not ln.strip().lower().startswith(("critique:", "## critique"))][:4]


def _revision_text(prompt: str, messages: Sequence[Message]) -> str:
    """Rewrite the last assistant draft so every critique bullet is addressed."""
    drafts = [m for m in list(messages)[:-1]
              if getattr(m, "role", "") == "assistant"
              and str(getattr(m, "content", "") or "").strip()]
    draft = str(drafts[-1].content) if drafts else prompt.strip()
    critique = _critique_block(prompt)
    crit_bullets = _critique_bullets(critique)

    draft_sections = _parse_sections(draft)
    out: list[str] = []
    for header in _REPORT_HEADERS:
        out.append(header)
        bullets = list(draft_sections.get(header, []))
        if header == "## Actions" and crit_bullets:
            bullets.extend(f"- addressed: {b}"[:200] for b in crit_bullets)
        out.extend(bullets if bullets else ["- none"])

    concrete = next((tok for tok in (_BACKTICK_RE.findall(critique) + _SRC_TOKEN_RE.findall(critique))
                     if tok not in draft and tok.lower() not in _NON_CONCRETE), "")
    if concrete and "## Evidence" in out:
        out.insert(out.index("## Evidence") + 1, f"- addresses critique point: {concrete}")

    out.append(f"Revised for: {(crit_bullets[0] if crit_bullets else 'critique feedback')[:120]}")
    out.append("Status: revised")

    if len("\n".join(out)) <= len(draft):
        pad_count = (len(draft) + 1 - len("\n".join(out))) // (len(_RISK_PAD) + 1) + 1
        idx = out.index("## Risks") + 1
        out[idx:idx] = [_RISK_PAD] * pad_count
    return "\n".join(out)


def _default_responder(
    llm: "MockLLM",
    call_index: int,
    agent: Any,
    messages: Sequence[Message],
    tools: "Sequence[ToolSpec]",
) -> LLMResponse:
    """Deterministic intent parser; priority documented at the module top."""
    del agent, tools
    prompt = _last_prompt(messages)
    conversation = _conversation_text(messages)
    observations = _observations_from_messages(messages)
    if observations:
        conversation = observations + "\n" + conversation
    marker_at = prompt.rfind(TOOL_CALL_MARKER)
    tool_request = marker_at != -1 or _TOOL_REQUEST_HEADER in prompt.lower()
    # Observations WIN over a leftover tool-request block: once tool results are
    # in the conversation the model must answer, not re-request the same files
    # (``--max-rounds`` would otherwise burn the budget on an identical request
    # and the draft would never become a report).
    if observations or (_has_observation(conversation) and not _has_critique(conversation)):
        source = observations or conversation
        text = _report_from_observations(source if observations else prompt, _is_round_zero(messages))
        if os.environ.get("OMP_MOCK_SCORE_STALL") == "1":
            with llm._lock:
                if llm._stall_first is None:
                    llm._stall_first = text
                else:
                    text = llm._stall_first
        return llm._response(text, [], call_index=call_index, messages=messages)
    if _has_critique(conversation):
        return llm._response(_revision_text(prompt, messages), [],
                             call_index=call_index, messages=messages)
    if tool_request:
        calls = parse_tool_calls(prompt) or _derive_tool_calls(prompt)
        if calls:
            return llm._response("", calls, call_index=call_index, messages=messages)
    return llm._response(_plan_text(prompt), [], call_index=call_index, messages=messages)


class MockLLM:
    """Deterministic offline provider. Zero network, zero credentials.

    ``responder(call_index, agent, messages, tools) -> LLMResponse`` replaces
    the default intent parser; ``agent``/``tools`` are duck-typed and may be
    ``None``. The call counter is guarded by a ``threading.Lock`` so parallel
    agents never collide; tool-call ids are stable per prompt (call_1, call_2…).
    """

    def __init__(
        self,
        *,
        name: str = "mock",
        responder: Optional[Callable[[int, Any, Sequence[Message], "Sequence[ToolSpec]"], LLMResponse]] = None,
        model: str = "mock-deterministic",
    ) -> None:
        """Configure the provider; ``responder=None`` uses the default parser."""
        self.name = name
        self.model = model
        self._responder = responder
        self._lock = threading.Lock()
        self._counter = 0
        self._stall_first: Optional[str] = None

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: "Sequence[ToolSpec]" = (),
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        agent: Any = None,
    ) -> LLMResponse:
        """Dispatch one call to the responder with a collision-free index."""
        with self._lock:
            call_index = self._counter
            self._counter += 1
        msgs = list(messages)
        if self._responder is None:
            return _default_responder(self, call_index, agent, msgs, tools)
        response = self._responder(call_index, agent, msgs, tools)
        # Normalize custom-responder output onto the mock contract.
        if response.provider != self.name:
            response.provider = self.name
        if not response.model:
            response.model = self.model
        if not response.usage:
            response.usage = self._usage(
                sum(len(str(getattr(m, "content", "") or "")) for m in msgs), response.text)
        if not response.raw.get("mock"):
            response.raw = {**response.raw, "mock": True, "call_index": call_index}
        return response

    def _usage(self, prompt_chars: int, text: str) -> dict[str, int]:
        """Approximate token counts (chars // 4), never wall-clock based."""
        return {"prompt_tokens": prompt_chars // 4,
                "completion_tokens": max(1, len(text) // 4) if text else 0}

    def _response(self, text: str, calls: list[ToolCall], *, call_index: int,
                  messages: Sequence[Message]) -> LLMResponse:
        """Wrap responder output with mock metadata and approximate usage."""
        prompt_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
        return LLMResponse(text=text, tool_calls=list(calls),
                           raw={"mock": True, "call_index": call_index},
                           model=self.model, usage=self._usage(prompt_chars, text),
                           provider=self.name)


def build_llm(
    *,
    strategy: Optional[LLMStrategy] = None,
    mock_responder: Optional[Callable] = None,
) -> tuple[LLM, str]:
    """Resolve the provider from OMP_LLM_MODE / strategy / credentials.

    ``mock`` -> ``(MockLLM, "mock")``. ``live`` -> real provider (or
    ``LLMConfigError`` when unconfigured). ``auto`` (default) -> real provider
    when configured, else ``MockLLM``. A strategy wins over the env var and is
    returned in its place; the mode still reflects whether a real provider is
    usable. Mode is always ``"mock"`` or ``"openai-compatible"``.
    """
    mode = (os.environ.get("OMP_LLM_MODE") or "auto").strip().lower()
    if mode == "mock":
        return (strategy if strategy is not None else MockLLM(responder=mock_responder)), "mock"
    if mode == "live":
        provider = OpenAICompatibleLLM.from_env()  # raises LLMConfigError when unconfigured
        return (strategy if strategy is not None else provider), "openai-compatible"
    try:
        resolved: LLM = OpenAICompatibleLLM.from_env()
        name = "openai-compatible"
    except LLMConfigError:
        resolved, name = MockLLM(responder=mock_responder), "mock"
    return (strategy if strategy is not None else resolved), name


class ProviderFallback:
    """LLMStrategy: primary until its first failure, then fallback.

    ``choose`` returns the primary object *itself* (so ``chosen is primary``
    holds); identity is preserved because the primary is instrumented in place
    rather than wrapped. ``__init__`` replaces the primary instance's
    ``complete`` with a recorder that, on the first exception, flips
    ``degraded`` and appends a sanitized, key-free warning to ``.warnings``
    before re-raising (the caller still surfaces the error). After that,
    ``choose`` returns ``fallback`` (or ``None`` when absent, meaning "let the
    agent surface the error"). ``report_failure`` lets a caller record the
    failure explicitly.
    """

    def __init__(
        self,
        primary: LLM,
        fallback: Optional[LLM] = None,
        *,
        warnings: Optional[list[str]] = None,
    ) -> None:
        """Configure the chain; ``warnings`` defaults to a fresh exposed list."""
        self.primary = primary
        self.fallback = fallback
        self.warnings: list[str] = warnings if warnings is not None else []
        self._failed = False
        self._lock = threading.Lock()
        self._instrument(primary)

    def _instrument(self, llm: LLM) -> None:
        """Wrap ``llm.complete`` in place; the object identity stays ``llm``."""
        if getattr(llm, "_squad_patched", False):
            return
        owner = self
        original = llm.complete

        def guarded(*args: Any, **kwargs: Any) -> LLMResponse:
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                owner.report_failure(exc)
                raise

        try:
            llm.complete = guarded  # type: ignore[assignment]
            llm._squad_patched = True  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # provider forbids instance attrs
            pass

    def report_failure(self, exc: BaseException) -> None:
        """Mark degraded once; append a truncated (<=200 chars) key-free warning."""
        message = f"provider {getattr(self.primary, 'name', '?')} failed: {type(exc).__name__}: {exc}"[:200]
        key = getattr(self.primary, "_api_key", None)
        if isinstance(key, str) and key and key in message:
            message = message.replace(key, "[redacted]")[:200]
        with self._lock:
            already = self._failed
            self._failed = True
        if not already:
            self.warnings.append(message)

    def choose(self, agent: Any, attempt: int) -> Optional[LLM]:
        """The primary object itself until its first failure, then fallback."""
        with self._lock:
            return self.fallback if self._failed else self.primary

    @property
    def active(self) -> str:
        """Name of the provider that ``choose`` would currently return."""
        with self._lock:
            current: Optional[LLM] = self.fallback if self._failed else self.primary
        return str(getattr(current, "name", "?")) if current is not None else "none"

    @property
    def degraded(self) -> bool:
        """True once the primary has failed at least once."""
        with self._lock:
            return self._failed