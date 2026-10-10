"""Blackboard: shared append-only message log + namespaced findings.

Coordination model (stigmergy): agents never talk to each other directly.
They publish Messages to a single append-only log and read each other's work
through filters (inbox/messages/thread) plus a key/value store partitioned
into namespaces.  Each agent owns exactly one findings namespace (e.g.
"security"), writes partial results there, and downstream agents pull them
via require().  This removes the need for pairwise channels and makes every
interaction inspectable/replayable from the log alone.

Locking: two threading.RLocks, never taken in the reverse order.
_log_lock guards the message list (publish, all read filters, clear).
_findings_lock guards the namespace dict (set/patch/findings/require/remove).
Lock discipline: log lock is ALWAYS acquired before the findings lock when
both are needed; the findings lock is never held while acquiring the log lock,
so no nested-order deadlock is possible.  RLock (not Lock) because public
helpers call each other while holding the lock.

Message.metadata free-form keys used across the system:
kind, thread, task_id, score, revisions, status, expects_reply,
tool_call_id, wave, group.

Canonical kind values (metadata["kind"]):
"assignment", "summary", "revision", "request", "reply", "observation",
"escalation", "report".

Deep-copy discipline: every read path returns copy.deepcopy(...) so callers
cannot mutate internal state, and write paths copy values on the way in.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

CANONICAL_KINDS = (
    "assignment",
    "summary",
    "revision",
    "request",
    "reply",
    "observation",
    "escalation",
    "report",
)


@dataclass
class ToolCall:
    """One requested tool invocation attached to a Message."""

    name: str
    arguments: dict[str, Any]
    id: str = ""  # runtime bookkeeping

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": copy.deepcopy(self.arguments), "id": self.id}


@dataclass
class Message:
    """A single unit of inter-agent communication."""

    sender: str  # agent name
    recipient: str = "*"  # agent name, or "*" for broadcast
    content: str = ""
    role: str = "assistant"  # "system" | "user" | "assistant" | "tool"
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0  # time.time() assigned at publish

    @property
    def kind(self) -> str:
        return str(self.metadata.get("kind", ""))

    def __str__(self) -> str:
        return f"[{self.role}] {self.sender} -> {self.recipient}: {self.content}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "role": self.role,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "metadata": copy.deepcopy(self.metadata),
            "timestamp": round(self.timestamp, 6),
        }


class Blackboard:
    """Shared workspace: append-only message log + namespaced key/value findings."""

    def __init__(self) -> None:
        self._log_lock = threading.RLock()
        self._findings_lock = threading.RLock()
        self._messages: list[Message] = []
        self._findings: dict[str, dict[str, Any]] = {}

    # -- publishing --------------------------------------------------------

    def publish(self, message: Message) -> None:
        """Append message to the log; stamp timestamp when falsy (0.0)."""
        with self._log_lock:
            if not message.timestamp:
                message.timestamp = time.time()
            self._messages.append(message)

    def emit(
        self,
        sender: str,
        content: str,
        *,
        kind: str = "",
        recipient: str = "*",
        **metadata: Any,
    ) -> Message:
        """Build a Message, stamp kind/metadata, publish, and return it."""
        msg = Message(sender=sender, recipient=recipient, content=content, metadata=dict(metadata))
        if kind:
            msg.metadata["kind"] = kind
        self.publish(msg)
        return msg

    def broadcast(self, sender: str, content: str, **metadata: Any) -> Message:
        return self.emit(sender, content, recipient="*", **metadata)

    def send(self, sender: str, recipient: str, content: str, **metadata: Any) -> Message:
        return self.emit(sender, content, recipient=recipient, **metadata)

    def ask(self, sender: str, recipient: str, content: str, **metadata: Any) -> Message:
        metadata["expects_reply"] = True
        return self.emit(sender, content, recipient=recipient, **metadata)

    # -- reading the log ---------------------------------------------------

    def messages(
        self,
        *,
        recipient: str | None = None,
        sender: str | None = None,
        role: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Filtered deep-copied snapshot; filters AND-combined, insertion order.

        limit returns only the LAST limit matches.
        """
        with self._log_lock:
            snapshot = list(self._messages)
        matched = [
            m
            for m in snapshot
            if (recipient is None or m.recipient == recipient)
            and (sender is None or m.sender == sender)
            and (role is None or m.role == role)
            and (kind is None or m.metadata.get("kind") == kind)
        ]
        if limit is not None:
            matched = matched[-limit:] if limit > 0 else []
        return [copy.deepcopy(m) for m in matched]

    def inbox(self, agent: str) -> list[Message]:
        """Messages addressed to agent (direct or broadcast), excluding own posts."""
        with self._log_lock:
            snapshot = list(self._messages)
        return [
            copy.deepcopy(m)
            for m in snapshot
            if m.sender != agent and (m.recipient == agent or m.recipient == "*")
        ]

    def thread(self, key: str) -> list[Message]:
        """Messages whose metadata['thread'] == key, insertion order."""
        with self._log_lock:
            snapshot = list(self._messages)
        return [copy.deepcopy(m) for m in snapshot if m.metadata.get("thread") == key]

    def since(self, index: int) -> list[Message]:
        """Messages appended after position index; out-of-range clamps to []."""
        with self._log_lock:
            n = len(self._messages)
            if index < 0 or index >= n:
                return []
            tail = self._messages[index:]
        return [copy.deepcopy(m) for m in tail]

    def cursor(self) -> int:
        """Current log length; pair with since() for incremental polling."""
        with self._log_lock:
            return len(self._messages)

    def transcript(self) -> list[Message]:
        """Full deep-copied snapshot of the log."""
        with self._log_lock:
            snapshot = list(self._messages)
        return [copy.deepcopy(m) for m in snapshot]

    def __len__(self) -> int:
        with self._log_lock:
            return len(self._messages)

    def __repr__(self) -> str:
        with self._log_lock:
            n = len(self._messages)
        return f"Blackboard(messages={n}, namespaces={self.namespaces()})"

    # -- namespaced findings ----------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Write a single key into the default '_shared' namespace."""
        self.patch("_shared", {key: value})

    def patch(self, namespace: str, values: Mapping[str, Any]) -> None:
        """Merge a deep-copied mapping into namespace (created if absent)."""
        with self._findings_lock:
            store = self._findings.setdefault(namespace, {})
            for k, v in values.items():
                store[k] = copy.deepcopy(v)

    def findings(self, namespace: str) -> dict[str, Any]:
        """Deep-copied namespace contents; unknown namespace returns {}."""
        with self._findings_lock:
            store = self._findings.get(namespace)
            if not store:
                return {}
            return copy.deepcopy(store)

    def require(self, namespace: str, *keys: str) -> dict[str, Any]:
        """Merged subset of namespace; ValueError names missing keys + namespaces."""
        with self._findings_lock:
            store = self._findings.get(namespace, {})
            missing = [k for k in keys if k not in store]
            if missing:
                available = self.namespaces()
                raise ValueError(
                    f"missing required findings from namespace {namespace!r}: {missing}; "
                    f"namespaces with data: {available}"
                )
            return {k: copy.deepcopy(store[k]) for k in keys}

    def remove(self, namespace: str) -> None:
        """Drop an entire namespace (no error if absent)."""
        with self._findings_lock:
            self._findings.pop(namespace, None)

    def namespaces(self) -> list[str]:
        """Sorted namespaces holding at least one key ('_shared' only if non-empty)."""
        with self._findings_lock:
            return sorted(ns for ns, store in self._findings.items() if store)

    def keys(self, namespace: str) -> list[str]:
        """Sorted key list for a namespace; [] when unknown."""
        with self._findings_lock:
            store = self._findings.get(namespace, {})
            return sorted(store)

    # -- inspection --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Timestamp-free summary, byte-stable across runs for assertions."""
        with self._log_lock:
            snapshot = list(self._messages)
        kinds: dict[str, int] = {}
        senders: dict[str, int] = {}
        for m in snapshot:
            k = str(m.metadata.get("kind", ""))
            kinds[k] = kinds.get(k, 0) + 1
            senders[m.sender] = senders.get(m.sender, 0) + 1
        return {
            "messages": len(snapshot),
            "namespaces": self.namespaces(),
            "kinds": {k: kinds[k] for k in sorted(kinds)},
            "senders": {s: senders[s] for s in sorted(senders)},
        }

    def dump(self) -> dict[str, Any]:
        """JSON-serialisable snapshot: works with json.dumps(..., sort_keys=True)."""
        with self._log_lock:
            messages = [m.to_dict() for m in self._messages]
        with self._findings_lock:
            findings = {ns: copy.deepcopy(store) for ns, store in self._findings.items() if store}
        return {"messages": messages, "findings": findings}

    def clear(self) -> None:
        """Reset log and findings (test isolation); swap containers under both locks."""
        with self._log_lock:
            with self._findings_lock:
                self._messages = []
                self._findings = {}

    # -- concurrency self-check -------------------------------------------

    def stress(self, writers: int = 8, per_writer: int = 50) -> dict[str, Any]:
        """Hammer publish + patch from N threads; returns counts and any errors."""
        errors: list[str] = []

        def worker(idx: int) -> None:
            try:
                for i in range(per_writer):
                    self.broadcast("stress-writer", f"msg {idx}-{i}", kind="observation")
                    self.patch(f"stress-{idx}", {f"key-{i}": i})
            except Exception as exc:  # noqa: BLE001 - surface in result, never raise
                errors.append(f"writer {idx}: {exc!r}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return {
            "messages": len(self),
            "namespaces": len(self.namespaces()),
            "errors": errors,
        }