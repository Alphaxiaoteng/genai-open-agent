"""Command line interface for the agent squad.

    python -m squad.cli demo            # offline, deterministic, no credentials
    python -m squad.cli audit <path>    # audit a real package
    python -m squad.cli version

Exit codes: ``0`` success, ``1`` the squad produced unfinished/failed work,
``2`` usage or configuration error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from squad.llm import LLMConfigError, build_llm
from squad.memory import Blackboard
from squad.squad import SquadResult
from squad.tools import build_default_registry
from squad.workflow import EventPrinter, build_task_text, collect_source_files, render_report

__all__ = ["main", "build_parser", "cmd_demo", "cmd_audit"]

EXIT_OK = 0
EXIT_UNFINISHED = 1
EXIT_USAGE = 2

VERSION = "0.1.0"


def add_run_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the shared run flags to ``parser``.

    ``argparse`` only binds optionals that appear AFTER the subcommand to the
    subparser, so the same flag set is attached to the top-level parser and to
    every subparser; both ``squad --quiet demo`` and ``squad demo --quiet``
    therefore work. ``SUPPRESS`` defaults keep the top-level values authoritative
    when a flag is given in both places.
    """
    parser.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="print per-agent output while running")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="emit the machine-readable report on stdout")
    parser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="suppress progress events on stderr")
    parser.add_argument("--workdir", default=argparse.SUPPRESS, help="workspace root for tool sandboxing")
    parser.add_argument("--max-rounds", type=int, default=argparse.SUPPRESS,
                        help="self-reflection rounds per agent (default 3)")
    parser.add_argument("--max-workers", type=int, default=argparse.SUPPRESS,
                        help="parallel agents per dependency wave")
    parser.add_argument("--focus", default=argparse.SUPPRESS,
                        help="extra instruction injected into every agent prompt")
    parser.add_argument("--repeat", type=int, default=argparse.SUPPRESS,
                        help="run the squad N times (determinism check)")
    parser.add_argument("--offline", action="store_true", default=argparse.SUPPRESS,
                        help="force the deterministic offline provider")
    parser.add_argument("--live", action="store_true", default=argparse.SUPPRESS,
                        help="require a real OpenAI-compatible provider")


#: Defaults applied after parsing when a run flag was given nowhere.
RUN_DEFAULTS: Dict[str, Any] = {
    "verbose": False,
    "json": False,
    "quiet": False,
    "workdir": None,
    "max_rounds": 3,
    "max_workers": 4,
    "focus": "",
    "repeat": 1,
    "offline": False,
    "live": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="squad",
        description="Multi-agent collaboration squad with self-reflection loops and closed-loop tool execution.",
    )
    parser.add_argument("--version", action="version", version="agent-squad %s" % VERSION)
    add_run_flags(parser)

    sub = parser.add_subparsers(dest="command")
    demo = sub.add_parser(
        "demo",
        help="audit a generated flawed sample package (offline, zero credentials)",
        parents=[],
    )
    add_run_flags(demo)
    demo.add_argument("--keep", action="store_true", help="keep the generated workspace and print its path")

    audit = sub.add_parser("audit", help="audit a real package directory")
    add_run_flags(audit)
    audit.add_argument("path", help="directory to audit")
    audit.add_argument("--ext", default="py,md", help="comma separated extensions to include (default py,md)")
    audit.add_argument("--out", default=None, help="write the rendered report to this file too")

    version = sub.add_parser("version", help="print the version and exit")
    add_run_flags(version)
    return parser


def apply_run_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in run-flag values that were suppressed because they were not given."""
    for key, value in RUN_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _resolve_mode(args: argparse.Namespace) -> Optional[str]:
    """Translate --offline/--live into OMP_LLM_MODE, rejecting contradictions."""
    if args.offline and args.live:
        raise SystemExit("--offline and --live are mutually exclusive")
    if args.offline:
        return "mock"
    if args.live:
        return "live"
    return None


def _prepare_provider(args: argparse.Namespace):
    mode = _resolve_mode(args)
    if mode is not None:
        os.environ["OMP_LLM_MODE"] = mode
    try:
        return build_llm()
    except LLMConfigError as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _run(squad, paths: List[str], args: argparse.Namespace, blackboard: Blackboard) -> SquadResult:
    from squad.workflow import run_audit

    printer = EventPrinter(enabled=not args.quiet, verbose=args.verbose)
    return run_audit(paths, squad=squad, blackboard=blackboard, focus=args.focus, on_event=printer)


def _emit(result: SquadResult, blackboard: Blackboard, args: argparse.Namespace, out_file: Optional[str]) -> int:
    report = render_report(result, blackboard=blackboard)
    if args.json:
        payload: Dict[str, Any] = {
            "report": result.to_dict(),
            "blackboard": blackboard.dump(),
        }
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(report)
    if out_file:
        with open(out_file, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")
        if not args.json:
            print("report written to %s" % out_file)
    return EXIT_OK if result.ok() else EXIT_UNFINISHED


def cmd_demo(args: argparse.Namespace) -> int:
    from squads.code_audit import build_code_audit_squad, sample_workspace

    provider, provider_mode = _prepare_provider(args)
    if provider_mode == "mock" and not args.quiet:
        print("provider: offline deterministic mock (no credentials, no network)", file=sys.stderr)

    root = args.workdir or tempfile.mkdtemp(prefix="agent-squad-demo-")
    paths = sample_workspace(root)
    registry = build_default_registry(workdir=root)
    blackboard = Blackboard()
    squad = build_code_audit_squad(
        blackboard=blackboard,
        llm=provider,
        tools=registry,
        workdir=root,
        max_rounds=args.max_rounds,
    )
    squad.max_workers = max(1, args.max_workers)

    code = EXIT_OK
    for attempt in range(max(1, args.repeat)):
        if args.repeat > 1 and not args.quiet:
            print("--- run %d/%d ---" % (attempt + 1, args.repeat), file=sys.stderr)
        blackboard.clear()
        result = _run(squad, paths, args, blackboard)
        code = _emit(result, blackboard, args, None)
        if _determinism_probe(args, attempt, result, blackboard):
            break
    if args.keep and not args.json:
        print("workspace: %s" % root)
    return code


def _determinism_probe(args: argparse.Namespace, attempt: int, result: SquadResult, board: Blackboard) -> bool:
    """With --repeat, compare runs and warn on drift (the demo's self-check)."""
    fingerprints = getattr(_determinism_probe, "_seen", None)
    current = {
        "scores": {name: round(r.score, 6) for name, r in result.results.items()},
        "statuses": {name: r.status for name, r in result.results.items()},
        "messages": board.stats().get("messages"),
    }
    if fingerprints is None:
        _determinism_probe._seen = current  # type: ignore[attr-defined]
        return False
    if fingerprints != current and not args.quiet:
        print("warning: run %d drifted from the previous run" % (attempt + 1), file=sys.stderr)
    return True


def cmd_audit(args: argparse.Namespace) -> int:
    from squads.code_audit import build_code_audit_squad

    target = os.path.abspath(args.path)
    if not os.path.isdir(target):
        print("error: not a directory: %s" % target, file=sys.stderr)
        return EXIT_USAGE

    provider, _ = _prepare_provider(args)
    exts = [e.strip().lower() for e in args.ext.split(",") if e.strip()]
    paths = collect_source_files(target, include=exts)
    if not paths:
        print("error: no auditable files (extensions: %s) under %s" % (", ".join(exts), target), file=sys.stderr)
        return EXIT_USAGE

    registry = build_default_registry(workdir=target)
    blackboard = Blackboard()
    squad = build_code_audit_squad(
        blackboard=blackboard, llm=provider, tools=registry, workdir=target, max_rounds=args.max_rounds
    )
    squad.max_workers = max(1, args.max_workers)
    result = _run(squad, paths, args, blackboard)
    return _emit(result, blackboard, args, args.out)


def cmd_version(_: argparse.Namespace) -> int:
    print("agent-squad %s (python %s)" % (VERSION, ".".join(str(v) for v in sys.version_info[:3])))
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = apply_run_defaults(parser.parse_args(argv))
    command = args.command or "demo"
    if command == "version":
        return cmd_version(args)
    if command == "audit":
        return cmd_audit(args)
    return cmd_demo(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
