"""Programmatic demo: run the code-audit squad without touching the CLI.

    python examples/demo.py                 # offline, deterministic
    python examples/demo.py --live          # use a real OpenAI-compatible provider
    python examples/demo.py --workspace out # audit your own directory

The script is import-safe (no side effects outside ``main``) so the test suite
can reuse :func:`run_demo`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from squad.llm import build_llm  # noqa: E402
from squad.memory import Blackboard  # noqa: E402
from squad.tools import build_default_registry  # noqa: E402
from squad.workflow import EventPrinter, collect_source_files, render_report, run_audit  # noqa: E402
from squads.code_audit import build_code_audit_squad, sample_workspace  # noqa: E402


def run_demo(
    workspace: Optional[str] = None,
    *,
    offline: bool = True,
    max_rounds: int = 3,
    quiet: bool = False,
    keep: bool = False,
) -> Dict[str, Any]:
    """Run the audit squad and return ``{"result", "blackboard", "report", "workspace"}``.

    ``offline=True`` forces the deterministic mock provider; the return value is
    JSON-serialisable through ``result.to_dict()`` and ``blackboard.dump()``.
    """
    if offline:
        os.environ["OMP_LLM_MODE"] = "mock"
    provider, mode = build_llm()

    root = workspace or tempfile.mkdtemp(prefix="agent-squad-demo-")
    sample_workspace(root)
    paths = collect_source_files(root, include=("py", "md"))

    registry = build_default_registry(workdir=root)
    blackboard = Blackboard()
    squad = build_code_audit_squad(blackboard=blackboard, llm=provider, tools=registry, workdir=root, max_rounds=max_rounds)
    printer = EventPrinter(enabled=not quiet)
    result = run_audit(paths, squad=squad, blackboard=blackboard, on_event=printer)

    payload = {
        "mode": mode,
        "workspace": root,
        "files": paths,
        "ok": result.ok(),
        "result": result.to_dict(),
        "blackboard": blackboard.dump(),
        "report": render_report(result, blackboard=blackboard),
    }
    if not keep and workspace is None:
        payload["workspace_retained"] = root  # temp dir left in place for inspection
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="agent-squad demo")
    parser.add_argument("--workspace", default=None, help="audit this directory instead of the generated sample")
    parser.add_argument("--live", action="store_true", help="use a real OpenAI-compatible provider")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="print the JSON payload instead of the report")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    payload = run_demo(args.workspace, offline=not args.live, max_rounds=args.max_rounds, quiet=args.quiet)
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(payload["report"])
        print("provider : %s" % payload["mode"])
        print("workspace: %s" % payload["workspace"])
    return 0 if payload["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
