"""Closed-loop tool execution for agent-squad.

The LLM proposes tool calls; this registry validates them under a sandbox
policy (timeout, output cap, dangerous gate, strict schema checking) and
converts the result into an observation fed back into the agent loop.

Trust boundary: every failure mode — unknown tool, missing/extra/badly typed
arguments, permission denial, dangerous gate, subprocess timeout, non-zero
exit, undecodable file, path escape, or unexpected crash — returns a
``ToolResult(ok=False, error="<CODE>: <message>")``. No exception escapes
``call()``, so the caller never needs a try/except around tool dispatch.

Error codes (machine-readable prefix for tests and report): NOT_FOUND,
MISSING_ARGS, EXTRA_ARGS, BAD_TYPE, DANGEROUS, PERMISSION, TIMEOUT, EXIT,
ERROR.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar
from urllib.parse import urlparse

F = TypeVar("F", bound=Callable[..., Any])

_SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "node_modules"})
_LARGE_FILE_BYTES = 1_048_576


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base class for all tool-level faults."""


class ToolNotFoundError(ToolError):
    """Requested tool name is not registered."""


class ToolArgumentError(ToolError):
    """Arguments are missing, extra, or fail type coercion."""


class ToolPermissionError(ToolError):
    """A policy gate (allowlist, dangerous flag, network) refused execution."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Immutable description of a tool exposed to agents."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema draft-07 object
    dangerous: bool = False


@dataclass
class ToolResult:
    """Outcome of one tool invocation; the object agents see as observation."""

    name: str
    ok: bool
    output: str = ""
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    exit_code: Optional[int] = None
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """Paired spec + implementation stored inside the registry."""

    spec: ToolSpec
    func: Callable[..., Any]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Thread-safe tool registry with sandboxed dispatch and an audit trail."""

    def __init__(self, *, allow_dangerous: bool = False) -> None:
        self._tools: Dict[str, Tool] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._audit: List[ToolResult] = []
        self._timeout: float = 60.0
        self._max_output: int = 100_000
        self.allow_dangerous: bool = allow_dangerous
        self.sandbox_limits: Dict[str, Any] = {}

    # -- registration -------------------------------------------------------

    def register(self, spec: ToolSpec, func: Callable[..., Any]) -> None:
        """Add a tool; re-registration of the same name overwrites silently."""
        with self._lock:
            self._tools[spec.name] = Tool(spec=spec, func=func)
            if spec.name not in self._order:
                self._order.append(spec.name)

    def tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        *,
        dangerous: bool = False,
    ) -> Callable[[F], F]:
        """Decorator that registers the wrapped function as a tool."""
        spec = ToolSpec(name=name, description=description, parameters=parameters, dangerous=dangerous)

        def decorator(fn: F) -> F:
            self.register(spec, fn)
            return fn

        return decorator

    # -- queries ------------------------------------------------------------

    def has(self, name: str) -> bool:
        """True if a tool with this name is registered."""
        return name in self._tools

    def specs(self) -> List[ToolSpec]:
        """Specs in registration order."""
        return [self._tools[n].spec for n in self._order]

    def summary(self) -> str:
        """One line per tool: name(param1, param2?) — description."""
        lines: List[str] = []
        for spec in self.specs():
            props = spec.parameters.get("properties", {})
            required = set(spec.parameters.get("required", []))
            parts: List[str] = []
            for pname in props:
                suffix = "" if pname in required else "?"
                parts.append(f"{pname}{suffix}")
            lines.append(f"{spec.name}({', '.join(parts)}) — {spec.description}")
        return "\n".join(lines)

    def describe(self) -> str:
        """Deterministic LLM-facing prompt block."""
        if not self._order:
            return "(no tools available)"
        sections: List[str] = []
        has_dangerous = False
        for spec in self.specs():
            schema_json = json.dumps(spec.parameters, sort_keys=True, ensure_ascii=False)
            sections.append(f"### {spec.name}\n{spec.description}\nparameters: {schema_json}")
            if spec.dangerous:
                has_dangerous = True
        block = "\n\n".join(sections)
        if has_dangerous:
            block += "\n\ndangerous tools require allow_dangerous=True"
        return block

    def describe_json(self) -> str:
        """Same information as describe(), serialized as JSON for --json."""
        items: List[Dict[str, Any]] = []
        for spec in self.specs():
            items.append({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "dangerous": spec.dangerous,
            })
        return json.dumps(items, indent=2, sort_keys=True, ensure_ascii=False)

    def audit_log(self) -> List[ToolResult]:
        """Snapshot of all results recorded by this registry."""
        with self._lock:
            return list(self._audit)

    # -- sandbox ------------------------------------------------------------

    def sandbox(self, *, timeout: float = 10.0, max_output: int = 8000) -> "ToolRegistry":
        """New registry sharing tool definitions, with independent limits/audit."""
        child = ToolRegistry()
        child._tools = self._tools
        child._order = self._order
        child._lock = threading.Lock()
        child._audit = []
        child._timeout = timeout
        child._max_output = max_output
        child.allow_dangerous = self.allow_dangerous
        child.sandbox_limits = {"timeout": timeout, "max_output": max_output}
        return child

    def __enter__(self) -> "ToolRegistry":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    # -- dispatch -----------------------------------------------------------

    def call(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        allow_dangerous: bool = False,
    ) -> ToolResult:
        """Invoke a tool by name. NEVER raises for tool faults."""
        args_dict: Dict[str, Any] = (
            dict(arguments) if isinstance(arguments, Mapping) else {"value": arguments}
        )

        try:
            tool = self._tools.get(name)
            if tool is None:
                available = ", ".join(self._order) if self._order else "none"
                return self._record(ToolResult(
                    name=name, ok=False,
                    error=f"NOT_FOUND: unknown tool '{name}'; available: {available}",
                    arguments=args_dict,
                ))

            spec = tool.spec

            # --- schema validation ---
            required = spec.parameters.get("required", [])
            missing = [k for k in required if k not in args_dict]
            if missing:
                names = ", ".join(repr(k) for k in missing)
                return self._record(ToolResult(
                    name=name, ok=False,
                    error=f"MISSING_ARGS: {names}",
                    arguments=args_dict,
                ))

            properties = spec.parameters.get("properties", {})
            extra = [k for k in args_dict if k not in properties]
            if extra:
                names = ", ".join(repr(k) for k in extra)
                return self._record(ToolResult(
                    name=name, ok=False,
                    error=f"EXTRA_ARGS: {names}",
                    arguments=args_dict,
                ))

            # --- type coercion ---
            try:
                args_dict = _coerce_types(args_dict, properties)
            except ToolArgumentError as exc:
                return self._record(ToolResult(
                    name=name, ok=False,
                    error=f"BAD_TYPE: {exc}",
                    arguments=args_dict,
                ))

            # --- dangerous gate ---
            if spec.dangerous and not (allow_dangerous or self.allow_dangerous):
                return self._record(ToolResult(
                    name=name, ok=False,
                    error=f"DANGEROUS: tool '{name}' is dangerous; pass allow_dangerous=True",
                    arguments=args_dict,
                ))

            # --- execute ---
            t0 = time.perf_counter()
            try:
                raw = tool.func(**args_dict)
            except ToolPermissionError as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return self._record(ToolResult(
                    name=name, ok=False, elapsed_ms=elapsed,
                    error=f"PERMISSION: {exc}",
                    arguments=args_dict,
                ))
            except subprocess.TimeoutExpired as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return self._record(ToolResult(
                    name=name, ok=False, elapsed_ms=elapsed,
                    error=f"TIMEOUT: {exc}",
                    arguments=args_dict,
                ))
            except ToolArgumentError as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                code = _classify_arg_error(str(exc))
                return self._record(ToolResult(
                    name=name, ok=False, elapsed_ms=elapsed,
                    error=f"{code}: {exc}",
                    arguments=args_dict,
                ))
            except ToolError as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return self._record(ToolResult(
                    name=name, ok=False, elapsed_ms=elapsed,
                    error=f"ERROR: {exc}",
                    arguments=args_dict,
                ))

            elapsed = (time.perf_counter() - t0) * 1000.0

            # --- normalize output ---
            if isinstance(raw, str):
                output = raw
            elif raw is None:
                output = ""
            else:
                output = json.dumps(raw, ensure_ascii=False, default=str)

            # --- truncate to registry cap ---
            limit = self._max_output
            if len(output) > limit:
                excess = len(output) - limit
                output = output[:limit] + f"\n... [truncated {excess} chars]"

            return self._record(ToolResult(
                name=name, ok=True, output=output, elapsed_ms=elapsed,
                arguments=args_dict,
            ))

        except Exception as exc:  # noqa: BLE001 — call() never raises
            return self._record(ToolResult(
                name=name, ok=False,
                error=f"ERROR: {exc}",
                arguments=args_dict,
            ))

    def call_many(
        self,
        calls: Sequence[Any],
        *,
        allow_dangerous: bool = True,
    ) -> List[ToolResult]:
        """Sequential dispatch; accepts ToolCall-likes or dicts."""
        results: List[ToolResult] = []
        for c in calls:
            if isinstance(c, Mapping):
                name = c.get("name", "")
                arguments = c.get("arguments", {})
            else:
                name = getattr(c, "name", "")
                arguments = getattr(c, "arguments", {})
            results.append(self.call(name, arguments, allow_dangerous=allow_dangerous))
        return results

    # -- internals ----------------------------------------------------------

    def _record(self, result: ToolResult) -> ToolResult:
        with self._lock:
            self._audit.append(result)
        return result


# ---------------------------------------------------------------------------
# Coercion and classification helpers
# ---------------------------------------------------------------------------


def _coerce_types(args: Dict[str, Any], properties: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in args.items():
        declared = properties.get(key, {}).get("type")
        if declared and isinstance(val, str):
            out[key] = _coerce(val, declared, key)
        else:
            out[key] = val
    return out


def _coerce(val: str, target: str, name: str) -> Any:
    if target == "integer":
        try:
            return int(val)
        except (ValueError, TypeError):
            raise ToolArgumentError(f"'{name}': cannot convert {val!r} to integer")
    if target == "number":
        try:
            return float(val)
        except (ValueError, TypeError):
            raise ToolArgumentError(f"'{name}': cannot convert {val!r} to number")
    if target == "boolean":
        low = val.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no", ""):
            return False
        raise ToolArgumentError(f"'{name}': cannot convert {val!r} to boolean")
    if target in ("array", "object"):
        try:
            parsed = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            raise ToolArgumentError(f"'{name}': cannot convert {val!r} to {target}")
        expected = list if target == "array" else dict
        if not isinstance(parsed, expected):
            raise ToolArgumentError(f"'{name}': JSON is not a {target}")
        return parsed
    return val  # string or unknown type: pass through


def _classify_arg_error(msg: str) -> str:
    low = msg.lower()
    if "timed out" in low:
        return "TIMEOUT"
    if "exited with code" in low:
        return "EXIT"
    return "ERROR"


# ---------------------------------------------------------------------------
# Path containment helper
# ---------------------------------------------------------------------------


def _resolve_contained(workdir: Path, user_path: str) -> Tuple[Path, str]:
    """Resolve user_path relative to workdir. Raises on sandbox escape."""
    if os.path.isabs(user_path):
        candidate = Path(user_path).resolve()
    else:
        candidate = (workdir / user_path).resolve()
    workdir_real = workdir.resolve()
    if candidate != workdir_real and not str(candidate).startswith(str(workdir_real) + os.sep):
        raise ToolArgumentError(f"path escapes the sandboxed workdir: {user_path}")
    rel = candidate.relative_to(workdir_real).as_posix()
    return candidate, rel


# ---------------------------------------------------------------------------
# Built-in tool builders
# ---------------------------------------------------------------------------


def read_file_tool(
    *,
    workdir: str | Path,
    max_bytes: int = 200_000,
) -> Tuple[ToolSpec, Callable[..., Any]]:
    """Build the read_file spec/callable for *workdir*."""
    wd = Path(workdir).resolve()

    def _read(*, path: str = "", max_bytes: int = max_bytes, **_kw: Any) -> str:
        if not path:
            raise ToolArgumentError("path is required")
        resolved, rel = _resolve_contained(wd, path)
        if not resolved.exists():
            raise ToolArgumentError(f"not a file: {path}")
        if resolved.is_dir():
            entries = sorted(p.name for p in resolved.iterdir())[:20]
            listing = "\n".join(f"  {e}" for e in entries)
            raise ToolArgumentError(f"is a directory: {path}\n{listing}")
        full_bytes = resolved.read_bytes()
        total = len(full_bytes)
        clipped = full_bytes[:max_bytes]
        truncated = total > max_bytes
        try:
            text = clipped.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            text = clipped.decode("latin-1")
            header = f"# {rel} (binary, {total} bytes)"
            body = f"{header}\n[binary] {text}"
            if truncated:
                body += f"\n... [truncated {total - max_bytes} bytes]"
            return body
        body = f"# {rel} ({total} bytes)\n{text}"
        if truncated:
            body += f"\n... [truncated {total - max_bytes} bytes]"
        return body

    spec = ToolSpec(
        name="read_file",
        description="Read a text file inside the sandboxed workdir.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workdir"},
                "max_bytes": {"type": "integer", "description": "Maximum bytes to read"},
            },
            "required": ["path"],
        },
    )
    return spec, _read


def write_file_tool(
    *,
    workdir: str | Path,
) -> Tuple[ToolSpec, Callable[..., Any]]:
    """Build the write_file spec/callable for *workdir*."""
    wd = Path(workdir).resolve()

    def _write(
        *,
        path: str = "",
        content: str = "",
        mkdir: bool = True,
        overwrite: bool = True,
        backup: bool = False,
        **_kw: Any,
    ) -> str:
        if not path:
            raise ToolArgumentError("path is required")
        resolved, rel = _resolve_contained(wd, path)
        if resolved.exists() and resolved.is_dir():
            raise ToolArgumentError(f"is a directory: {path}")
        if resolved.exists() and not overwrite:
            raise ToolArgumentError(f"file exists: {path}")
        if mkdir:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        if backup and resolved.exists():
            bak = resolved.with_name(resolved.name + ".bak")
            bak.write_bytes(resolved.read_bytes())
        encoded = content.encode("utf-8")
        resolved.write_bytes(encoded)
        return f"wrote {rel} ({len(encoded)} bytes)"

    spec = ToolSpec(
        name="write_file",
        description="Write content to a file inside the sandboxed workdir.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workdir"},
                "content": {"type": "string", "description": "Content to write (UTF-8)"},
                "mkdir": {"type": "boolean", "description": "Create parent directories (default true)"},
                "overwrite": {"type": "boolean", "description": "Overwrite existing file (default true)"},
                "backup": {"type": "boolean", "description": "Create .bak before overwriting (default false)"},
            },
            "required": ["path", "content"],
        },
        dangerous=True,
    )
    return spec, _write


def list_files_tool(
    *,
    workdir: str | Path,
    max_entries: int = 200,
) -> Tuple[ToolSpec, Callable[..., Any]]:
    """Build the list_files spec/callable for *workdir*."""
    wd = Path(workdir).resolve()

    def _list(*, pattern: str = "**/*", max_entries: int = max_entries, **_kw: Any) -> str:
        all_paths = sorted(wd.glob(pattern), key=lambda p: p.as_posix())
        total = len(all_paths)
        shown = all_paths[:max_entries]
        lines: List[str] = []
        for p in shown:
            try:
                lines.append(p.relative_to(wd).as_posix())
            except ValueError:
                lines.append(p.as_posix())
        if total > max_entries:
            lines.append(f"... [truncated {total - max_entries} entries]")
        lines.append(f"# {total} entries")
        return "\n".join(lines)

    spec = ToolSpec(
        name="list_files",
        description="List files matching a glob pattern inside the workdir.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (default **/*)"},
                "max_entries": {"type": "integer", "description": "Max paths to return"},
            },
            "required": [],
        },
    )
    return spec, _list


def search_text_tool(
    *,
    workdir: str | Path,
    max_matches: int = 100,
) -> Tuple[ToolSpec, Callable[..., Any]]:
    """Build the search_text (literal substring) spec/callable for *workdir*."""
    wd = Path(workdir).resolve()

    def _search(*, query: str = "", path: str = ".", max_matches: int = max_matches, **_kw: Any) -> str:
        if not query:
            raise ToolArgumentError("query is required")
        root = (wd / path).resolve()
        skipped_binary = 0
        skipped_large = 0
        matches: List[str] = []
        files_with_match: set = set()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                if size > _LARGE_FILE_BYTES:
                    skipped_large += 1
                    continue
                try:
                    head = fpath.read_bytes()[:4096]
                except OSError:
                    continue
                if b"\x00" in head:
                    skipped_binary += 1
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    if query in line:
                        try:
                            rel = fpath.relative_to(wd).as_posix()
                        except ValueError:
                            rel = fpath.as_posix()
                        excerpt = line.strip()[:200]
                        matches.append(f"{rel}:{lineno}: {excerpt}")
                        files_with_match.add(rel)
                        if len(matches) >= max_matches:
                            break
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break
        lines = matches + [
            f"# {len(matches)} matches in {len(files_with_match)} files",
            f"# skipped: {skipped_binary} binary, {skipped_large} large",
        ]
        return "\n".join(lines)

    spec = ToolSpec(
        name="search_text",
        description="Search for a literal substring across files in the workdir.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Literal text to search for"},
                "path": {"type": "string", "description": "Subdirectory to search (default .)"},
                "max_matches": {"type": "integer", "description": "Max matches to return"},
            },
            "required": ["query"],
        },
    )
    return spec, _search


def python_exec_tool(
    *,
    workdir: str | Path,
    timeout: float = 10.0,
    max_output: int = 8000,
) -> Tuple[ToolSpec, Callable[..., Any]]:
    """Build the python_exec spec/callable running code via subprocess."""
    wd = Path(workdir).resolve()

    def _exec(*, code: str = "", timeout: float = timeout, max_output: Optional[int] = None, **_kw: Any) -> str:
        if not code:
            raise ToolArgumentError("code is required")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(wd),
                env=None,
            )
        except subprocess.TimeoutExpired:
            raise ToolArgumentError(f"python timed out after {timeout}s")
        except OSError as exc:
            raise ToolArgumentError(str(exc))
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[:2000]
            raise ToolArgumentError(f"python exited with code {proc.returncode}: {stderr_tail}")
        out = proc.stdout
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        if max_output is not None and len(out) > max_output:
            excess = len(out) - max_output
            out = out[:max_output] + f"\n... [truncated {excess} chars]"
        return out

    spec = ToolSpec(
        name="python_exec",
        description="Execute Python code in a subprocess inside the sandboxed workdir.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute"},
                "timeout": {"type": "number", "description": "Subprocess timeout in seconds"},
                "max_output": {"type": "integer", "description": "Override the output cap for this call"},
            },
            "required": ["code"],
        },
        dangerous=True,
    )
    return spec, _exec


def shell_command_tool(
    *,
    workdir: str | Path,
    allowlist: Sequence[str] = ("python", "python3", "ls", "cat", "wc", "echo", "git"),
    timeout: float = 10.0,
    max_output: int = 8000,
) -> Tuple[ToolSpec, Callable[..., Any]]:
    """Build the shell_command spec/callable with an allowlist gate."""
    wd = Path(workdir).resolve()
    _META = (";", "|", "&&", "`", "$(")

    def _shell(*, command: str = "", timeout: float = timeout, max_output: Optional[int] = None, **_kw: Any) -> str:
        if not command:
            raise ToolArgumentError("command is required")
        for meta in _META:
            if meta in command:
                raise ToolPermissionError(
                    f"command contains forbidden metacharacter '{meta}'; chaining is not allowed"
                )
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ToolArgumentError(f"invalid quoting in command: {exc}")
        if not parts:
            raise ToolArgumentError("empty command")
        prog = os.path.basename(parts[0])
        if prog not in allowlist:
            raise ToolPermissionError(
                f"command '{prog}' is not in the allowlist: {sorted(allowlist)}"
            )
        try:
            proc = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(wd),
                shell=False,
                env=None,
            )
        except subprocess.TimeoutExpired:
            raise ToolArgumentError(f"command timed out after {timeout}s")
        except OSError as exc:
            raise ToolArgumentError(str(exc))
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[:2000]
            raise ToolArgumentError(f"command exited with code {proc.returncode}: {stderr_tail}")
        out = proc.stdout
        if max_output is not None and len(out) > max_output:
            excess = len(out) - max_output
            out = out[:max_output] + f"\n... [truncated {excess} chars]"
        return out

    spec = ToolSpec(
        name="shell_command",
        description="Run an allowlisted shell command (single command, no chaining).",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run (must be allowlisted)"},
                "timeout": {"type": "number", "description": "Subprocess timeout in seconds"},
                "max_output": {"type": "integer", "description": "Override the output cap for this call"},
            },
            "required": ["command"],
        },
        dangerous=True,
    )
    return spec, _shell


def http_get_tool(
    *,
    timeout: float = 10.0,
    max_bytes: int = 100_000,
    allow_hosts: Optional[Sequence[str]] = None,
) -> Tuple[ToolSpec, Callable[..., Any]]:
    """Build the http_get spec/callable with optional host allowlist."""
    builder_hosts = allow_hosts

    def _get(*, url: str = "", timeout: float = timeout, max_bytes: int = max_bytes,
             allow_hosts: Optional[Sequence[str]] = None, **_kw: Any) -> str:
        if not url:
            raise ToolArgumentError("url is required")
        # Builder policy is authoritative; a per-call allow_hosts may only narrow.
        hosts = builder_hosts if builder_hosts is not None else allow_hosts
        if hosts is not None:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            matched = any(host == h or host.endswith("." + h) for h in hosts)
            if not matched:
                raise ToolPermissionError(
                    f"host '{host}' is not in the allowlist: {sorted(hosts)}"
                )
        req = urllib.request.Request(url, headers={"User-Agent": "agent-squad/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                body = resp.read(max_bytes)
        except urllib.error.HTTPError as exc:
            raise ToolArgumentError(f"HTTP {exc.code}: {exc.reason}")
        except OSError as exc:
            raise ToolArgumentError(f"network error: {exc}")
        if status < 200 or status >= 300:
            raise ToolArgumentError(f"non-2xx response: {status}")
        return body.decode("utf-8", errors="replace")

    spec = ToolSpec(
        name="http_get",
        description="Fetch a URL with HTTP GET and return the response body.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "timeout": {"type": "number", "description": "Request timeout in seconds"},
                "max_bytes": {"type": "integer", "description": "Max response bytes to return"},
                "allow_hosts": {"type": "array", "description": "Host allowlist (may only narrow the builder policy)"},
            },
            "required": ["url"],
        },
        dangerous=True,
    )
    return spec, _get


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------

_DEFAULT_SHELL_ALLOWLIST = ("python", "python3", "ls", "cat", "wc", "echo", "git")


def build_default_registry(
    *,
    workdir: str | Path,
    timeout: float = 10.0,
    max_output: int = 8000,
) -> ToolRegistry:
    """Pre-loaded registry: read/write/list/search/python_exec/shell_command/http_get.

    Creates *workdir* when missing. This is the framework's own execution
    policy: every tool is callable inside the configured sandbox without an
    extra per-call flag (``allow_dangerous=True``), while workdir containment,
    the command allowlist and the resource caps still apply. Callers that want
    the stricter default build a bare ``ToolRegistry()``, which returns
    ``DANGEROUS:`` for gated tools unless ``allow_dangerous=True`` is passed.
    """
    wd = Path(workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)

    reg = ToolRegistry(allow_dangerous=True)
    reg._timeout = timeout
    reg._max_output = max_output
    reg.sandbox_limits = {"timeout": timeout, "max_output": max_output}

    builders: List[Tuple[ToolSpec, Callable[..., Any]]] = [
        read_file_tool(workdir=wd, max_bytes=200_000),
        write_file_tool(workdir=wd),
        list_files_tool(workdir=wd, max_entries=200),
        search_text_tool(workdir=wd, max_matches=100),
        python_exec_tool(workdir=wd, timeout=timeout, max_output=max_output),
        shell_command_tool(workdir=wd, allowlist=_DEFAULT_SHELL_ALLOWLIST, timeout=timeout, max_output=max_output),
        http_get_tool(timeout=timeout, max_bytes=100_000, allow_hosts=None),
    ]
    for spec, fn in builders:
        reg.register(spec, fn)

    return reg