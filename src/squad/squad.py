"""Squad orchestration: stigmergy over a shared blackboard + a dependency DAG.

The algorithm
-------------

- Agents are organised into named *groups*.  A group declares ``depends_on``
  other groups, which builds a DAG at group level.  ``plan()`` runs Kahn's
  algorithm over that DAG with ``sorted()`` tie-breaking, producing a stable
  list of *waves* (each wave is a list of mutually independent groups).  Two
  consecutive calls to ``plan()`` on the same squad return identical output.

- ``run()`` executes one wave at a time.  Every agent of a wave is submitted to
  a single ``ThreadPoolExecutor(max_workers=min(max_workers, len(wave)))`` and
  consumed with ``as_completed``.  Coordination is stigmergic — agents talk only
  through the shared blackboard, never through return values — so the only
  ordering guarantee between agents is the wave barrier: wave N+1 starts once
  every wave-N agent has settled.

- Wave isolation is what makes runs reproducible: a sibling in the same wave
  MUST NOT be treated as a dependency.  An agent's execution context (its
  ``needs`` findings and ``upstream_summaries``) is assembled from the blackboard
  the moment the agent actually starts.  With ``max_workers == 1`` each wave is
  submitted in alphabetical order, so a later same-wave agent may observe an
  earlier same-wave sibling in ``sibling_outputs`` — that sequential-execution
  bonus is intentional and documented, and parallel runs stay order-independent
  because results are always keyed by agent name.

- Per-agent failure isolation: any ``Exception`` raised by ``execute()`` is
  converted into a failed ``AgentResult`` inside the worker so one bad agent can
  never abort the squad; the other futures still run to completion.
  ``KeyboardInterrupt``/``SystemExit`` are never swallowed (only ``Exception`` is
  caught), and a wave's executor is always shut down in a ``finally``.

- Event contract for UIs (``on_event``): every callback is wrapped in a
  try/except so a broken listener cannot abort the run.  Events are plain dicts:
  ``{"type": "group", "wave", "group", "status", "elapsed_ms"}``,
  ``{"type": "agent", "wave", "group", "agent", "status"}`` for starts, and
  ``{"type": "result", ...}`` carrying ``score``/``elapsed_ms`` and a truncated
  ``output`` for each completion.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from squad.agent import Agent, AgentResult
from squad.memory import Blackboard

SIBLING_OUTPUT_MAX = 500
EVENT_OUTPUT_MAX = 300
EXCERPT_MAX = 400


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` clipped to ``limit`` characters (no ellipsis churn)."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit]


@dataclass
class SquadResult:
    """Aggregate outcome of a single ``Squad.run()``."""

    name: str
    rounds: List[Dict[str, Any]]
    results: "OrderedDict[str, AgentResult]"
    elapsed_ms: float = 0.0

    def ok(self) -> bool:
        """True iff the squad ran at least one agent and every status is "ok"."""
        return bool(self.results) and all(
            r.status == "ok" for r in self.results.values()
        )

    @property
    def scores(self) -> Dict[str, float]:
        """Agent name -> final score."""
        return {name: r.score for name, r in self.results.items()}

    def __getitem__(self, key: str) -> AgentResult:
        try:
            return self.results[key]
        except KeyError:
            available = ", ".join(self.results) or "none"
            raise KeyError(
                f"agent '{key}' not in squad '{self.name}' (available: {available})"
            ) from None

    def __len__(self) -> int:
        return len(self.results)

    def __contains__(self, key: str) -> bool:
        return key in self.results

    def __iter__(self) -> Iterator[AgentResult]:
        return iter(self.results.values())

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot; ``OrderedDict``s become plain dicts."""
        return {
            "squad": self.name,
            "rounds": self.rounds,
            "elapsed_ms": self.elapsed_ms,
            "ok": self.ok(),
            "scores": self.scores,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


class Squad:
    """Dependency-DAG coordinator over a set of :class:`Agent` members."""

    def __init__(
        self,
        name: str,
        *,
        blackboard: Optional[Blackboard] = None,
        max_workers: int = 4,
        name_prefix: Optional[str] = None,
    ) -> None:
        self.name = name
        self.blackboard = blackboard
        self.max_workers = max(1, int(max_workers))
        if name_prefix is not None:
            self.name_prefix = name_prefix
        else:
            self.name_prefix = os.environ.get("SQUAD_AGENT_PREFIX", "") or ""
        self._agents: "OrderedDict[str, Agent]" = OrderedDict()
        self._groups: "OrderedDict[str, List[str]]" = OrderedDict()
        self._depends: Dict[str, Tuple[str, ...]] = {}
        self._needs: Dict[str, Tuple[str, ...]] = {}
        self._objectives: "OrderedDict[str, str]" = OrderedDict()
        self._prompts: "OrderedDict[str, str]" = OrderedDict()

    # -- registration ------------------------------------------------------

    def add(
        self,
        agent: Agent,
        *,
        group: str = "default",
        depends_on: Tuple[str, ...] = (),
        needs: Tuple[str, ...] = (),
        objective: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Agent:
        """Register ``agent`` under ``group``; returns the agent (chainable)."""
        # Apply the squad prefix only when it is set and not already present, so
        # an agent that arrives pre-prefixed (e.g. from build_code_audit_squad)
        # is never renamed twice.
        if self.name_prefix and not agent.name.startswith(self.name_prefix):
            old_name = agent.name
            agent.name = self.name_prefix + old_name
            if getattr(agent, "write_namespace", None) == old_name:
                agent.write_namespace = agent.name
        if agent.name in self._agents:
            raise ValueError(f"agent '{agent.name}' is already registered")
        setattr(agent, "squad", self)
        self._agents[agent.name] = agent
        members = self._groups.setdefault(group, [])
        if agent.name not in members:
            members.append(agent.name)
        self._depends[group] = tuple(dict.fromkeys(depends_on))
        self._needs[agent.name] = tuple(dict.fromkeys(needs))
        resolved_objective = objective or getattr(agent, "goal", "") or agent.name
        self._objectives[agent.name] = resolved_objective
        self._prompts[agent.name] = prompt or resolved_objective
        return agent

    def add_agents(self, agents: List[Agent], **kwargs: Any) -> List[Agent]:
        """Register every agent in ``agents`` with the same keyword options."""
        return [self.add(agent, **kwargs) for agent in agents]

    # -- introspection -----------------------------------------------------

    def groups(self) -> List[str]:
        """Group names in registration order."""
        return list(self._groups)

    def members(self, group: Optional[str] = None) -> List[Agent]:
        """Agents of ``group`` (all agents when ``group`` is None)."""
        if group is None:
            return list(self._agents.values())
        return [self._agents[name] for name in self._groups.get(group, [])]

    def objectives(self) -> Dict[str, str]:
        """Agent name -> objective (the default task prompt)."""
        return dict(self._objectives)

    def group_dependencies(self) -> Dict[str, List[str]]:
        """Group name -> declared ``depends_on`` groups."""
        return {g: list(deps) for g, deps in self._depends.items()}

    def find(self, name: str) -> Agent:
        """Look up a member by name; KeyError lists the available agents."""
        try:
            return self._agents[name]
        except KeyError:
            available = ", ".join(self._agents) or "none"
            raise KeyError(
                f"agent '{name}' not found in squad '{self.name}' (available: {available})"
            ) from None

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __repr__(self) -> str:
        return (
            f"Squad(name='{self.name}', groups={self.groups()}, "
            f"agents={len(self._agents)})"
        )

    # -- planning ----------------------------------------------------------

    def plan(self) -> List[List[str]]:
        """Deterministic DAG waves via Kahn over groups (sorted tie-breaking).

        A group joins a wave only once every group it ``depends_on`` has been
        emitted, so ``depends_on`` edges point group -> dependency.  Unknown
        dependencies and cycles raise :class:`ValueError`.
        """
        for group, deps in self._depends.items():
            for dep in deps:
                if dep not in self._groups:
                    raise ValueError(
                        f"group '{dep}' (needed by '{group}') is not registered"
                    )
        remaining: set = set(self._groups)
        pending: Dict[str, set] = {
            g: set(self._depends.get(g, ())) & remaining for g in remaining
        }
        waves: List[List[str]] = []
        ready = sorted(g for g in remaining if not pending[g])
        while remaining:
            if not ready:
                raise ValueError(
                    f"dependency cycle among groups: {sorted(remaining)}"
                )
            wave = sorted(ready)
            waves.append(wave)
            remaining.difference_update(wave)
            emitted = set(wave)
            ready = []
            for group in sorted(remaining):
                pending[group] -= emitted
                if not pending[group]:
                    ready.append(group)
        return waves

    def summary(self) -> str:
        """Stable human-readable plan: one line per group per wave."""
        lines: List[str] = []
        for wave_index, wave in enumerate(self.plan(), start=1):
            for group in wave:
                agents = ", ".join(sorted(self._groups.get(group, [])))
                lines.append(f"wave {wave_index}: [{group}] {agents}")
        return "\n".join(lines)

    # -- event emission ----------------------------------------------------

    def _emit(self, on_event: Any, event: Dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:
            pass

    # -- context assembly --------------------------------------------------

    def _build_context(
        self,
        agent: Agent,
        *,
        focus: str,
        group: str,
        wave: int,
        blackboard: Blackboard,
        completed: Dict[str, str],
    ) -> Dict[str, Any]:
        """Snapshot the blackboard for ``agent`` (called at execution time)."""
        namespaces = self._needs.get(agent.name, ())
        needs: Dict[str, Dict[str, Any]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        summary_messages = blackboard.messages(kind="summary") if namespaces else []
        for namespace in namespaces:
            needs[namespace] = dict(blackboard.findings(namespace))
            chosen = None
            for message in summary_messages:
                sender = message.sender
                if sender == namespace or sender.startswith(namespace + ":"):
                    chosen = message
            if chosen is None:
                summaries[namespace] = {
                    "sender": "",
                    "score": None,
                    "content_excerpt": "",
                }
            else:
                summaries[namespace] = {
                    "sender": chosen.sender,
                    "score": chosen.metadata.get("score"),
                    "content_excerpt": _truncate(chosen.content, EXCERPT_MAX),
                }
        return {
            "focus": focus,
            "group": group,
            "wave": wave,
            "needs": needs,
            "upstream_summaries": summaries,
            "sibling_outputs": dict(completed),
        }

    def _failed_result(self, agent_name: str, prompt: str, warning: str) -> AgentResult:
        return AgentResult(
            agent=agent_name,
            task=prompt,
            status="failed",
            output="",
            score=0.0,
            scores={},
            revisions=0,
            reflections=[],
            tool_results=[],
            writes={},
            warnings=[warning],
            elapsed_ms=0.0,
            provider="",
        )

    def _run_agent(
        self,
        agent: Agent,
        prompt: str,
        *,
        board: Blackboard,
        focus: str,
        group: str,
        wave: int,
        task_id: str,
        completed: Dict[str, str],
        lock: threading.Lock,
    ) -> AgentResult:
        """Worker body: build context, execute, never raise past Exception."""
        begin = time.perf_counter()
        with lock:
            context = self._build_context(
                agent,
                focus=focus,
                group=group,
                wave=wave,
                blackboard=board,
                completed=completed,
            )
        try:
            return agent.execute(
                prompt, context=context, blackboard=board, task_id=task_id
            )
        except Exception as exc:
            return AgentResult(
                agent=agent.name,
                task=prompt,
                status="failed",
                output="",
                score=0.0,
                scores={},
                revisions=0,
                reflections=[],
                tool_results=[],
                writes={},
                warnings=[f"unhandled error: {type(exc).__name__}: {exc}"],
                elapsed_ms=(time.perf_counter() - begin) * 1000.0,
                provider="",
            )

    # -- execution ---------------------------------------------------------

    def run(
        self,
        *,
        focus: str = "",
        blackboard: Optional[Blackboard] = None,
        on_event: Optional[Any] = None,
    ) -> SquadResult:
        """Execute the planned waves; return results keyed by agent name.

        Unless a blackboard is passed explicitly, every run gets a *fresh* one
        so repeated ``run()`` calls (the CLI ``--repeat`` flow) never leak state
        between attempts.  The squad caches no agent state itself: ``writes`` are
        reset by each ``execute()`` call.
        """
        # Board resolution order: the run argument wins, then the board the
        # squad was constructed with, then a fresh one. Ignoring the
        # construction-time board would silently discard every finding a
        # caller wired up in advance.
        board = blackboard if blackboard is not None else self.blackboard
        if board is None:
            board = Blackboard()
        self.blackboard = board
        task_id = "focus" if focus else ""
        if focus:
            board.broadcast("orchestrator", focus, kind="assignment", task_id=task_id)

        results: "OrderedDict[str, AgentResult]" = OrderedDict()
        rounds: List[Dict[str, Any]] = []
        completed: Dict[str, str] = {}
        lock = threading.Lock()
        started_at = time.perf_counter()

        for wave_index, wave in enumerate(self.plan(), start=1):
            wave_size = sum(len(self._groups.get(g, [])) for g in wave)
            group_started = {g: time.perf_counter() for g in wave}
            remaining_in_group = {g: len(self._groups.get(g, [])) for g in wave}
            executor = ThreadPoolExecutor(
                max_workers=min(self.max_workers, max(1, wave_size)),
                thread_name_prefix=f"squad-{self.name}-w{wave_index}",
            )
            futures: "Dict[Future, Tuple[str, str, str]]" = {}
            try:
                for group in wave:
                    self._emit(
                        on_event,
                        {
                            "type": "group",
                            "wave": wave_index,
                            "group": group,
                            "status": "started",
                            "elapsed_ms": 0.0,
                        },
                    )
                    for name in sorted(self._groups.get(group, [])):
                        agent = self._agents[name]
                        # The latest prompt wins: a caller may re-target a squad
                        # for a new workspace (``workflow.run_audit``) after
                        # registration, so ``agent.prompt`` overrides the
                        # registration-time prompt when it is set.
                        prompt = getattr(agent, "prompt", None) or self._prompts[name]
                        self._emit(
                            on_event,
                            {
                                "type": "agent",
                                "wave": wave_index,
                                "group": group,
                                "agent": name,
                                "status": "started",
                            },
                        )
                        future = executor.submit(
                            self._run_agent,
                            agent,
                            prompt,
                            board=board,
                            focus=focus,
                            group=group,
                            wave=wave_index,
                            task_id=task_id,
                            completed=completed,
                            lock=lock,
                        )
                        futures[future] = (group, name, prompt)

                # Settle the wave; as_completed is a generator so KeyboardInterrupt
                # /SystemExit propagate instead of being swallowed here.
                for future in as_completed(futures):
                    group, name, prompt = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = self._failed_result(
                            name,
                            prompt,
                            f"unhandled error: {type(exc).__name__}: {exc}",
                        )
                    results[name] = result
                    with lock:
                        completed[name] = _truncate(result.output, SIBLING_OUTPUT_MAX)
                    self._emit(
                        on_event,
                        {
                            "type": "result",
                            "wave": wave_index,
                            "group": group,
                            "agent": name,
                            "status": result.status,
                            "score": result.score,
                            "elapsed_ms": result.elapsed_ms,
                            "output": _truncate(result.output, EVENT_OUTPUT_MAX),
                        },
                    )
                    remaining_in_group[group] -= 1
                    if remaining_in_group[group] == 0:
                        self._emit(
                            on_event,
                            {
                                "type": "group",
                                "wave": wave_index,
                                "group": group,
                                "status": "finished",
                                "elapsed_ms": (
                                    time.perf_counter() - group_started[group]
                                ) * 1000.0,
                            },
                        )
            finally:
                executor.shutdown(wait=False)
            for group in wave:
                group_agents = sorted(self._groups.get(group, []))
                rounds.append(
                    {
                        "wave": wave_index,
                        "group": group,
                        "agents": group_agents,
                        "elapsed_ms": (
                            time.perf_counter() - group_started[group]
                        ) * 1000.0,
                        "scores": {
                            name: results[name].score
                            for name in group_agents
                            if name in results
                        },
                    }
                )

        return SquadResult(
            name=self.name,
            rounds=rounds,
            results=results,
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
        )