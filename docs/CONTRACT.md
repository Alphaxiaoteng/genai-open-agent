# Interface Contract — frozen

Pure stdlib. `from __future__ import annotations` at top of every module. No third-party imports anywhere in `src/`.

## squad/memory.py — messages + shared blackboard

```python
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""                     # runtime bookkeeping

@dataclass
class Message:
    sender: str                      # agent name
    recipient: str = "*"             # agent name, or "*" for broadcast
    content: str = ""
    role: str = "assistant"          # "system" | "user" | "assistant" | "tool"
    tool_calls: list[ToolCall] = []
    metadata: dict[str, Any] = {}
    timestamp: float = 0.0           # time.time() at publish

class Blackboard:
    """Shared workspace: append-only message log + namespaced key/value findings."""
    def __init__(self) -> None
    def publish(self, message: Message) -> None
    def broadcast(self, sender: str, content: str, **metadata: Any) -> Message
    def send(self, sender: str, recipient: str, content: str, **metadata: Any) -> Message
    def ask(self, sender: str, recipient: str, content: str, **metadata: Any) -> Message  # metadata["expects_reply"]=True
    def messages(self, *, recipient: str | None = None, sender: str | None = None,
                 role: str | None = None, kind: str | None = None) -> list[Message]
    def inbox(self, agent: str) -> list[Message]     # broadcast + direct-to-agent, excluding own posts
    def transcript(self) -> list[Message]
    def set(self, key: str, value: Any) -> None      # low-level
    def patch(self, namespace: str, values: Mapping[str, Any]) -> None
    def findings(self, namespace: str) -> dict[str, Any]
    def require(self, namespace: str, *keys: str) -> dict[str, Any]  # ValueError if missing
    def namespaces(self) -> list[str]
    def keys(self, namespace: str) -> list[str]
    def __len__(self) -> int                         # message count
```

Kind filter: `Message.metadata["kind"]` — canonical kinds are `"assignment"`, `"summary"`, `"revision"`, `"request"`, `"reply"`, `"observation"`, `"escalation"`, `"report"`.

## squad/tools.py — closed-loop tool execution

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]                 # JSON Schema object
    dangerous: bool = False

@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str = ""
    error: str | None = None
    elapsed_ms: float = 0.0
    exit_code: int | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

class ToolError(Exception): ...
class ToolNotFoundError(ToolError): ...
class ToolArgumentError(ToolError): ...
class ToolPermissionError(ToolError): ...

@dataclass
class Tool:
    spec: ToolSpec
    func: Callable[..., Any]

class ToolRegistry:
    def __init__(self) -> None
    def register(self, spec: ToolSpec, func: Callable[..., Any]) -> None
    def tool(self, name: str, description: str, parameters: dict[str, Any],
             *, dangerous: bool = False) -> Callable[[F], F]      # decorator
    def specs(self) -> list[ToolSpec]
    def describe(self) -> str                 # LLM prompt block, deterministic order
    def has(self, name: str) -> bool
    def call(self, name: str, arguments: Mapping[str, Any] | None = None,
             *, allow_dangerous: bool = False) -> ToolResult      # never raises for tool faults
    def sandbox(self, *, timeout: float = 10.0, max_output: int = 8000) -> ToolRegistry
    # context manager: with registry.sandbox(timeout=5) as sb: sb.call(...)
    def __enter__(self) -> ToolRegistry
    def __exit__(self, *exc) -> bool
    def audit_log(self) -> list[ToolResult]
```

`arguments` only contains OPTIONAL kwargs (positional params are described in `parameters` but never passed as kwargs — tools MUST take every parameter as keyword-or-default).

Built-in builders (in `squad/tools.py`):
```python
def build_default_registry(*, workdir: str | Path, timeout: float = 10.0, max_output: int = 8000) -> ToolRegistry
def read_file_tool(*, workdir, max_bytes=200_000) -> tuple[ToolSpec, Callable]
def write_file_tool(*, workdir) -> tuple[ToolSpec, Callable]
def list_files_tool(*, workdir, max_entries=200) -> tuple[ToolSpec, Callable]
def search_text_tool(*, workdir, max_matches=100) -> tuple[ToolSpec, Callable]  # literal, no regex
def python_exec_tool(*, timeout=10.0, max_output=8000) -> tuple[ToolSpec, Callable]
def shell_command_tool(*, allowlist, timeout=10.0, max_output=8000) -> tuple[ToolSpec, Callable]
def http_get_tool(*, timeout=10.0, max_bytes=100_000, allow_hosts=None) -> tuple[ToolSpec, Callable]
```

Tool names: `read_file`, `write_file`, `list_files`, `search_text`, `python_exec`, `shell_command`, `http_get`. `write_file` / `shell_command` / `python_exec` are `dangerous=True`.
Guarantee: every failure (`ToolNotFoundError`, `ToolArgumentError`, `ToolPermissionError`, bad args, crash, timeout, non-zero exit) returns `ToolResult(ok=False, error=...)`; NO exception escapes `call()`.

## squad/llm.py — provider abstraction

```python
@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    provider: str = ""

class LLMError(RuntimeError): ...
class LLMConfigError(LLMError): ...

class LLM(Protocol):
    name: str
    def complete(self, messages: Sequence[Message], *, tools: Sequence[ToolSpec] = (),
                 system: str | None = None, temperature: float = 0.2,
                 max_tokens: int = 1024) -> LLMResponse

class LLMStrategy(Protocol):
    def choose(self, agent: Agent, attempt: int) -> LLM

class OpenAICompatibleLLM:      # real HTTP, stdlib urllib
    def __init__(self, *, model, api_key=None, base_url=None, timeout=60.0, max_retries=2,
                 transport=None, provider="openai-compatible") -> None
    @classmethod
    def from_env(cls, **overrides) -> OpenAICompatibleLLM   # LLMConfigError if no key

class MockLLM:
    """Deterministic offline provider. Zero network, zero credentials."""
    def __init__(self, *, name="mock", responder=None, model="mock-deterministic") -> None
    # `responder(call_index: int, agent: Agent, messages, tools) -> LLMResponse`
    # default: intent parsing — numbered JSON tool block, tool-result observation → summary,
    #          "critique:" line → revision text.

def build_llm(*, strategy=None, mock_responder=None) -> tuple[LLM, str]
    # env OMP_BASE_URL / OMP_API_KEY / OMP_MODEL / OMP_LLM_MODE
    # no usable credentials → (MockLLM(...), "mock") ; else (OpenAICompatibleLLM, "openai-compatible")

class ProviderFallback:
    """Tries primary; on LLMError switches to fallback and records a warning on the agent."""
    def __init__(self, primary: LLM, fallback: LLM | None = None, *, warnings: list[str] | None = None) -> None
    def choose(self, agent: Agent, attempt: int) -> LLM
    @property
    def active(self) -> str
    @property
    def degraded(self) -> bool   # True once primary failed at least once
```

Env contract: `OMP_BASE_URL` (default `https://api.openai.com/v1`), `OMP_API_KEY` (fallback `OPENAI_API_KEY`), `OMP_MODEL` (default `gpt-4o-mini`), `OMP_LLM_MODE` ∈ {`auto`, `mock`, `live`} (`live` ⇒ raise `LLMConfigError` when unconfigured). HTTP via `urllib.request`; retries with backoff on 429/5xx/URLError; never logs the API key. `transport(request_dict, timeout) -> response_dict` injectable for tests.

## squad/agent.py — agent + self-reflection loop

```python
@dataclass
class ReflectionRound:
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
    def revised(self) -> bool
    def to_dict(self) -> dict[str, Any]

class ReflectionPolicy:
    min_score: float = 0.75
    min_dimension: float = 0.5
    max_rounds: int = 3
    scoring: Literal["deterministic", "llm"] = "deterministic"
    def __init__(self, *, min_score=0.75, min_dimension=0.5, max_rounds=3) -> None
    def score(self, output: str, *, task: str, tools_used: Sequence[str],
              tools_available: Sequence[str], revision: int) -> tuple[float, dict[str, float]]
    def evaluate(self, round_result, history) -> tuple[bool, str]   # (accepted, critique_or_reason)

@dataclass
class AgentResult:
    agent: str
    task: str
    status: str                 # "ok" | "unfinished" | "failed"
    output: str
    score: float
    scores: dict[str, float]
    revisions: int
    reflections: list[ReflectionRound]
    tool_results: list[ToolResult]
    writes: dict[str, Any]      # namespace -> values (mirrors self.writes)
    warnings: list[str]
    elapsed_ms: float
    provider: str
    def to_dict(self, *, include_reflections=True) -> dict[str, Any]

class Agent:
    role: str
    goal: str
    name: str
    def __init__(self, name, *, role="", goal="", instructions="", llm=None,
                 tools=None, policy=None, max_tool_calls=6, write_namespace=None,
                 output_key="output", tool_allow_dangerous=True, name_prefix=None) -> None
    @property writes: dict[str, Any]
    def resolve_llm(self) -> LLM
    def system_prompt(self) -> str
    def execute(self, task: str, *, context: Mapping[str, Any] | None = None,
                blackboard: Blackboard | None = None, task_id: str = "") -> AgentResult
    def observe(self) -> list[Message]      # from blackboard inbox
```

`context` is flattened into the prompt as `## Context` (`key: value` lines; lists/dicts via `json.dumps(ensure_ascii=False)`).
`execute()` loop per reflection round: (1) `complete()` → text + tool calls; (2) replay `agent`/`tool` message pairs and remaining budget; (3) `policy.evaluate` → accept or inject critique and re-prompt. Exactly one `complete()` call per round. Publish to blackboard: `kind="summary"` broadcast on accept, `kind="summary"` + `"unfinished"`/`"failed"` metadata otherwise. Status `"failed"` when the provider raised on round 0 and no output exists; otherwise `"unfinished"`.
Namespace defaults to the agent name; on accept the blackboard is patched with `output_key → output`, `score`, `scores`, `revisions`, `status`. `writes` dict has the same content.
`name_prefix` (or `SQUAD_AGENT_PREFIX` env) prefixes every child agent name so one squad can be instantiated twice in a process without blackboard/registry clashes.

## squad/squad.py — orchestration

```python
@dataclass
class SquadResult:
    name: str
    rounds: list[dict[str, Any]]
    results: OrderedDict[str, AgentResult]
    elapsed_ms: float
    def ok(self) -> bool
    def __getitem__(self, key: str) -> AgentResult
    def to_dict(self) -> dict[str, Any]

class Squad:
    def __init__(self, name: str, *, blackboard=None, max_workers=4, name_prefix=None) -> None
    def add(self, agent: Agent, *, group: str = "default", depends_on=(), needs=(),
            objective: str | None = None, prompt: str | None = None) -> Agent   # returns agent
    def groups(self) -> list[str]
    def members(self, group=None) -> list[Agent]
    def objectives(self) -> dict[str, str]
    def group_dependencies(self) -> dict[str, list[str]]
    def plan(self) -> list[list[str]]        # DAG waves; ValueError on unknown group/cycle
    def run(self, *, focus: str = "", blackboard=None, on_event=None) -> SquadResult
    def find(self, name: str) -> Agent
    def __iter__ -> Iterator[Agent]
    def __len__ -> int
```

- Group-level dependencies: all agents in a group run after the groups in `depends_on`.
- Waves execute in a `ThreadPoolExecutor(max_workers)`; per-agent `blackboard.inbox(agent.name)` must be visible at wave start ⇒ shared blackboard guarded by its own lock, and `Squad` calls `blackboard.flush()` (or none) — publish is immediate, so no flush needed. `run()` MUST keep identical results for `max_workers=1` and `max_workers=4` (deterministic tests compare membership + scores, not timings).
- Intra-wave ordering only via declared `depends_on`; agents in the same group MUST NOT read each other's writes. Use separate groups for real dependencies.
- `on_event(event: dict)` receives `{"type": "group"/"agent", "wave": int, "group": str, "agent": str, "status": str, "score": float, "elapsed_ms": float}`.

## squads/code_audit.py — the code-quality audit squad

```python
DEFAULT_SAMPLE_FILES: dict[str, str]      # flawed python snippets used by the offline demo/CI

def build_code_audit_squad(*, blackboard=None, llm=None, strategy=None, tools=None,
                           group="audit", name_prefix=None, max_rounds=3) -> Squad
```
Groups/roles: `architecture` (Architect) → `security` (SecuritySniffer) → `tests` (TestGenerator) → `review` (LeadAuditor). Every agent needs `read_file`/`list_files`/`search_text`; TestGenerator additionally needs `write_file` + `python_exec`. LeadAuditor reads writes from the three upstream namespaces via `needs=(...)`.

Offline determinism: `MockLLM` default responder parses the `## Task` block. The audit agents MUST emit `TOOL_CALLS` blocks derived from the task text (they may not invent file names), so a run over `DEFAULT_SAMPLE_FILES` is reproducible and every named acceptance criterion (see ARCHITECTURE.md) is satisfied.

## cli.py — `python -m squad.cli`

Subcommands: `demo` (build sample package into `--workdir`, run squad, print report + JSON), `audit <path>`, `version`.
Flags: `--offline` (force MockLLM), `--live` (require real provider), `--json`, `--workdir PATH`, `--max-rounds N`, `-v`.
Exit codes: `0` success, `1` squad produced `unfinished`/`failed` results, `2` usage error.

## Determinism / QA rules

- No `time.time()` inside scoring or assertions; timing fields are informational only.
- Every module runnable as `python -m squad.<module>`? No — only `squad.cli`.
- Tests must not require network or credentials.
