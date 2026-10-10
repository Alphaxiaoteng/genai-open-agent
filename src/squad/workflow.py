"""Audit workflow: orchestration glue between the squad and a target workspace.

The CLI and the end-to-end test share this module so the demo and the test
exercise the *same* path: materialise a workspace, build the squad, run it with
the audit task block, and render a report.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, TextIO

from squad.memory import Blackboard
from squad.squad import Squad, SquadResult
from squad.tools import ToolRegistry

__all__ = [
    "EventPrinter",
    "build_task_text",
    "collect_source_files",
    "render_report",
    "run_audit",
]


def collect_source_files(root: str, *, include: Sequence[str] = ("py", "md")) -> List[str]:
    """Return auditable files as relative POSIX paths, sorted.

    Only the extensions in ``include`` are returned; vendored and generated
    directories are skipped so the audit never drowns in third-party code.
    """
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".agent-squad"}
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for name in sorted(filenames):
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in include:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def build_task_text(paths: Sequence[str], *, focus: str = "", objective: str = "") -> str:
    """Compose the deterministic per-agent task block.

    ``## Tool request`` is the machine-readable section a provider may turn into
    tool calls; ``paths`` is echoed verbatim so a run is reproducible.
    """
    lines: List[str] = []
    if objective:
        lines.append("Objective: %s" % objective)
    if focus:
        lines.append("Focus: %s" % focus)
    lines.append("Audited package files (relative to the workspace): %s" % ", ".join(paths))
    lines.append("")
    lines.append("## Tool request")
    lines.append("read every audited file, then report; use the tools, do not guess.")
    lines.append("")
    lines.extend("- read %s" % p for p in paths)
    lines.append("- search for the pattern `eval(`")
    lines.append("- list files matching **/*.py")
    lines.append("- write tests/test_audit_regression.py")
    lines.append("- run the generated verification snippet")
    lines.append("")
    lines.append("## Required sections")
    lines.extend("- %s" % h for h in (
        "## Summary",
        "## Findings",
        "## Evidence",
        "## Actions",
        "## Verification",
        "## Risks",
    ))
    return "\n".join(lines)


class EventPrinter:
    """Streams squad events to a terminal as a compact progress log."""

    def __init__(self, stream: Optional[TextIO] = None, *, enabled: bool = True, verbose: bool = False) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        self.verbose = verbose
        self.waves = 0

    def __call__(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        kind = event.get("type")
        if kind == "group":
            if event.get("status") == "started":
                self.waves += 1
                self._write("wave %s -> group %s" % (event.get("wave"), event.get("group")))
            else:
                self._write("group %s done in %.0f ms" % (event.get("group"), event.get("elapsed_ms", 0.0)))
        elif kind == "agent":
            self._write("  %-28s %s" % (event.get("agent"), event.get("status")))
        elif kind == "result":
            line = "  -> %-26s score=%.3f %s" % (
                event.get("agent"),
                event.get("score", 0.0),
                event.get("status"),
            )
            if self.verbose:
                line += "\n     %s" % str(event.get("output", "")).replace("\n", "\n     ")[:400]
            self._write(line)

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text + "\n")
            self.stream.flush()
        except Exception:  # pragma: no cover - a broken stream must not kill a run
            self.enabled = False


def run_audit(
    paths: Sequence[str],
    *,
    squad: Squad,
    blackboard: Optional[Blackboard] = None,
    focus: str = "",
    on_event: Optional[Any] = None,
) -> SquadResult:
    """Run every squad member against ``paths`` and return the raw result.

    Agents are run through :meth:`Squad.run`; the per-agent task text is injected
    by setting ``prompt`` on each member so one squad definition serves any
    workspace.
    """
    board = blackboard if blackboard is not None else squad.blackboard
    for agent in squad:
        objective = squad.objectives().get(agent.name, agent.goal or agent.name)
        agent.prompt = build_task_text(paths, focus=focus, objective=objective)
        agent.task = agent.prompt
    return squad.run(focus=focus, blackboard=board, on_event=on_event)


def render_report(result: SquadResult, *, blackboard: Optional[Blackboard] = None) -> str:
    """Render a human-readable audit report from a finished squad run."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("AGENT SQUAD AUDIT REPORT")
    lines.append("=" * 72)
    lines.append("squad      : %s" % result.name)
    lines.append("status     : %s" % ("PASS" if result.ok() else "NEEDS WORK"))
    lines.append("elapsed    : %.0f ms" % result.elapsed_ms)
    lines.append("agents     : %d" % len(result.results))
    lines.append("")
    lines.append("WAVES")
    for wave in result.rounds:
        lines.append(
            "  wave %-2s %-14s %s" % (wave.get("wave"), wave.get("group"), ", ".join(wave.get("agents", [])))
        )
    lines.append("")
    lines.append("AGENTS")
    for name, res in result.results.items():
        lines.append("-" * 72)
        lines.append("%s  [%s]  score=%.3f  revisions=%d  tools=%d  provider=%s"
                     % (name, res.status, res.score, res.revisions, len(res.tool_results), res.provider))
        if res.scores:
            lines.append("  dimensions: " + "  ".join("%s=%.2f" % kv for kv in sorted(res.scores.items())))
        for warning in res.warnings:
            lines.append("  ! %s" % warning)
        tools_used = ", ".join(t.name for t in res.tool_results) or "none"
        lines.append("  tool calls : %s" % tools_used)
        for line in (res.output or "(no output)").splitlines():
            lines.append("  | " + line)
    if blackboard is not None:
        lines.append("")
        lines.append("BLACKBOARD")
        stats = blackboard.stats()
        lines.append("  messages   : %d" % stats.get("messages", 0))
        lines.append("  namespaces : %s" % ", ".join(stats.get("namespaces", [])))
        lines.append("  findings   : %s" % json.dumps(blackboard.dump().get("findings", {}), sort_keys=True)[:1200])
    lines.append("=" * 72)
    return "\n".join(lines)


def timed() -> float:
    """Monotonic timestamp helper shared by the CLI timings."""
    return time.perf_counter()
