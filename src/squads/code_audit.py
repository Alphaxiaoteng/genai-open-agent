"""Code-quality audit squad: an architecture / security / test closed loop.

This module assembles four specialised agents into one collaboration squad and
is the reference workload shipped with the framework:

* ``architecture``  Architect       - module boundaries, coupling, dead code, README/docs gaps
* ``security``      SecuritySniffer - injection sinks, unsafe deserialisation, secret leakage
* ``tests``         TestEngineer    - writes real regression tests into the workspace and RUNS them
* ``review``        LeadAuditor     - merges the three artefacts and gates the release

Coordination contract (see ``docs/ARCHITECTURE.md``):

1. Each upstream agent writes its findings into its own blackboard namespace
   (``<group-prefixed agent name>``) so a downstream agent can ``require(...)``
   them instead of re-reading the same files.
2. The ``tests`` group runs *after* ``security`` because a generated regression
   test should pin the vulnerability the sniffer found - not an inventory of
   style nits.
3. The ``review`` group consumes all three namespaces and emits the release
   gate. It never reads raw source files it does not need; its evidence comes
   from the upstream artefacts (stigmergic hand-off).

Everything is deterministic: the squad works with the offline ``MockLLM`` (no
credentials, no network) and with a real OpenAI-compatible endpoint without any
code change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from squad.agent import Agent, ReflectionPolicy
from squad.llm import LLM, LLMStrategy, build_llm
from squad.memory import Blackboard
from squad.squad import Squad
from squad.tools import ToolRegistry, build_default_registry

__all__ = [
    "DEFAULT_SAMPLE_FILES",
    "AUDIT_HEADERS",
    "build_code_audit_squad",
    "sample_workspace",
]

#: Section headers every audit artefact must carry; the reflection scorer
#: measures their presence, so they are defined once and reused in prompts.
AUDIT_HEADERS: Tuple[str, ...] = (
    "## Summary",
    "## Findings",
    "## Evidence",
    "## Actions",
    "## Verification",
    "## Risks",
)

#: A small, deliberately flawed Python package used by the one-click demo and
#: by the end-to-end test. Every flaw is real and verifiable by the tools:
#:   * ``db.find_user``            -> SQL injection sink (f-string interpolation)
#:   * ``auth.verify_token``       -> ``==`` secret comparison + hardcoded secret
#:   * ``auth.load_session``       -> ``eval`` on untrusted payload
#:   * ``billing.total``           -> mutable default argument + float money math
#:   * ``billing.parse_amount``    -> bare ``except`` swallowing errors
#:   * ``billing.charge``          -> unchecked external call, no retry
DEFAULT_SAMPLE_FILES: Dict[str, str] = {
    "db.py": '''"""Tiny data-access layer (deliberately flawed demo code)."""

CONNECTION = None


def connect(dsn):
    global CONNECTION
    CONNECTION = dsn
    return dsn


def find_user(username):
    sql = "SELECT * FROM users WHERE name = '%s'" % username
    if CONNECTION is None:
        raise RuntimeError("database not connected")
    return run_sql(sql)


def run_sql(sql):
    # Pretend the driver executed it; returns the statement for inspection.
    return {"sql": sql, "rows": []}


def delete_user(user_id):
    return run_sql("DELETE FROM users WHERE id = %s" % user_id)
''',
    "auth.py": '''"""Session and token handling (deliberately flawed demo code)."""

import hashlib

SECRET = "s3cr3t-hardcoded-signing-key"


def hash_password(password, salt=None):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_token(token):
    return token == SECRET


def load_session(payload):
    try:
        return eval(payload)
    except Exception:
        return {}
''',
    "billing.py": '''"""Billing helpers (deliberately flawed demo code)."""

import json

TAX = 0.19


def total(items=[]):
    amount = 0.0
    for item in items:
        amount = amount + item["price"] * item.get("qty", 1)
    return amount + amount * TAX


def parse_amount(raw):
    try:
        return float(raw)
    except:
        return 0.0


def charge(account, amount):
    if amount <= 0:
        return {"ok": True, "charged": 0}
    response = json.dumps({"account": account, "amount": amount})
    return {"ok": True, "charged": amount, "receipt": response}
''',
    "README.md": """# sample-package

A tiny billing/auth/data-access package used as the audit input.

## Usage

```python
from db import connect, find_user
connect("sqlite://demo")
print(find_user("alice"))
```
""",
}

_PERSONAS: Dict[str, Dict[str, str]] = {
    "architecture": {
        "role": "Architecture Reviewer",
        "goal": "map the module boundaries and find structural debt",
        "instructions": (
            "Read every module listed in the task, then report: coupling between "
            "layers, duplicated logic, dead or unreachable code, missing type "
            "hints/docstrings, and missing operational documentation. Cite the "
            "exact file path and line for every finding."
        ),
    },
    "security": {
        "role": "Security Sniffer",
        "goal": "find exploitable sinks and secret leakage in the audited package",
        "instructions": (
            "Read every module listed in the task and search for dangerous "
            "patterns (SQL string interpolation, eval/exec, unsafe "
            "deserialisation, hardcoded secrets, weak comparison, unchecked "
            "input). For each finding give: sink, file:line, exploit sketch, and "
            "the concrete fix. Never report a finding you did not see in a file."
        ),
    },
    "tests": {
        "role": "Test Engineer",
        "goal": "turn the audit findings into executable regression tests",
        "instructions": (
            "Using the security findings in the context, write REAL unittest "
            "files into the workspace with the write_file tool, then execute "
            "them with the python_exec tool and report the observed result. "
            "Every test must name the defect it pins."
        ),
    },
    "review": {
        "role": "Lead Auditor",
        "goal": "gate the release on the merged audit artefacts",
        "instructions": (
            "The three upstream artefacts are provided in the context under "
            "`needs`. Merge them into one release-gate report: what blocks the "
            "release, what is merely risky, the ordered action list, and the "
            "exact verification command for each action. Quote the upstream "
            "evidence; do not re-read files you were not asked to read."
        ),
    },
}


def sample_workspace(root: Any) -> List[str]:
    """Materialise :data:`DEFAULT_SAMPLE_FILES` under ``root``.

    Returns the sorted list of written relative paths. Existing files are
    overwritten - callers own the directory (the CLI creates a fresh temp dir).
    """
    import os

    os.makedirs(root, exist_ok=True)
    written: List[str] = []
    for rel, content in sorted(DEFAULT_SAMPLE_FILES.items()):
        path = os.path.join(root, rel)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        written.append(rel)
    return sorted(written)


def _files_block(paths: Sequence[str]) -> str:
    """Render the deterministic ``## Tool request`` block the agents emit."""
    lines = [
        "## Tool request",
        "read every source file listed below, then report your findings.",
        "",
    ]
    lines.extend("- read %s" % path for path in paths)
    lines.append("- search for the pattern `eval(`")
    lines.append("- list files matching **/*.py")
    lines.append("- write tests/test_audit_regression.py")
    lines.append("- run the generated verification snippet")
    return "\n".join(lines)


def build_code_audit_squad(
    *,
    blackboard: Optional[Blackboard] = None,
    llm: Optional[LLM] = None,
    strategy: Optional[LLMStrategy] = None,
    tools: Optional[ToolRegistry] = None,
    workdir: Optional[str] = None,
    group: str = "audit",
    name_prefix: Optional[str] = None,
    max_rounds: int = 3,
    max_agents: Optional[int] = None,
) -> Squad:
    """Assemble the code-quality audit squad.

    ``group`` is the namespace prefix for the four agents, which allows multiple
    squads (e.g. one per repository) to coexist inside a single blackboard.
    ``tools`` defaults to ``build_default_registry(workdir)``; ``workdir``
    defaults to ``$SQUAD_WORKSPACE`` or ``.agent-squad/workspace``.
    """
    import os

    workspace = workdir or os.environ.get("SQUAD_WORKSPACE") or os.path.join(".agent-squad", "workspace")
    registry = tools if tools is not None else build_default_registry(workdir=workspace)

    if llm is not None:
        provider: Optional[LLM] = llm
        mode = "explicit"
    else:
        provider, mode = build_llm(strategy=strategy)

    prefix = name_prefix if name_prefix is not None else os.environ.get("SQUAD_AGENT_PREFIX", "")
    squad_name = "%s-code-audit[%s]" % (group, mode)

    squad = Squad(squad_name, blackboard=blackboard, name_prefix=prefix, max_workers=4)

    def make(key: str) -> Agent:
        persona = _PERSONAS[key]
        return Agent(
            "%s-%s" % (group, key),
            role=persona["role"],
            goal=persona["goal"],
            instructions=persona["instructions"],
            llm=provider,
            tools=registry,
            policy=ReflectionPolicy(min_score=0.75, min_dimension=0.5, max_rounds=max_rounds),
            max_tool_calls=8 if key == "tests" else 6,
            tool_allow_dangerous=(key == "tests"),
        )

    architect = squad.add(
        make("architecture"),
        group="architecture",
        objective="map module boundaries and structural debt of the audited package",
    )
    sniffer = squad.add(
        make("security"),
        group="security",
        depends_on=("architecture",),
        objective="find exploitable sinks and secret leakage in the audited package",
    )
    engineer = squad.add(
        make("tests"),
        group="tests",
        depends_on=("security",),
        needs=(sniffer.write_namespace,),
        objective="write and execute regression tests for the confirmed vulnerabilities",
    )
    lead = squad.add(
        make("review"),
        group="review",
        depends_on=("architecture", "security", "tests"),
        needs=(
            architect.write_namespace,
            sniffer.write_namespace,
            engineer.write_namespace,
        ),
        objective="merge the audit artefacts into a single release gate",
    )

    if max_agents is not None:
        squad.max_agents = max_agents
    return squad


def squad_task_block(paths: Sequence[str], *, focus: str = "") -> str:
    """Build the per-round task text handed to the audit agents.

    The block is intentionally machine-readable: the offline ``MockLLM`` derives
    its tool calls from the ``## Tool request`` section, which is why the same
    run is reproducible without credentials.
    """
    parts: List[str] = []
    if focus:
        parts.append(focus.strip())
    parts.append("Audit these files: %s" % ", ".join(paths))
    parts.append(_files_block(paths))
    return "\n".join(parts)
