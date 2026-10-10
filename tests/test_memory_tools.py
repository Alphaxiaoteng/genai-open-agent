"""Blackboard + sandboxed tool registry tests."""

from __future__ import annotations

import json
import os
import threading
import unittest

from support import TempWorkspace

from squad.memory import Blackboard, Message, ToolCall
from squad.tools import ToolRegistry, ToolSpec, build_default_registry


class TestBlackboard(TempWorkspace):
    def test_messages_and_kinds(self) -> None:
        board = Blackboard()
        board.broadcast("a", "hello", kind="summary", score=0.9)
        board.send("a", "b", "direct", kind="reply")
        board.ask("a", "b", "question", kind="request")

        self.assertEqual(len(board), 3)
        self.assertEqual([m.content for m in board.messages(kind="summary")], ["hello"])
        self.assertEqual(board.messages(kind="reply")[0].recipient, "b")
        self.assertTrue(board.messages(kind="request")[0].metadata["expects_reply"])
        self.assertEqual(board.messages(sender="a", recipient="b")[0].content, "direct")
        self.assertEqual(board.transcript()[0].timestamp > 0, True)

    def test_kind_and_kitchen_sink(self) -> None:
        """kind travels through metadata; the emitter stamps the timestamp."""
        message = Message(sender="x", content="c")
        self.assertEqual(message.kind, "")
        self.assertTrue(json.dumps(message.to_dict(), sort_keys=True))

    def test_inbox_excludes_own_posts_and_direct_messages(self) -> None:
        board = Blackboard()
        board.broadcast("a", "to everyone")
        board.send("a", "b", "for b")
        board.send("a", "c", "for c")
        inbox = board.inbox("b")
        self.assertEqual(sorted(m.content for m in inbox), ["for b", "to everyone"])
        self.assertEqual(board.inbox("z")[0].content, "to everyone")

    def test_namespaces_isolated_and_deep_copied(self) -> None:
        board = Blackboard()
        board.patch("security", {"critical": 3, "items": ["sql", "eval"]})
        board.patch("architecture", {"modules": 4})
        board.set("shared_key", 1)

        self.assertEqual(board.namespaces(), ["_shared", "architecture", "security"])
        self.assertEqual(board.keys("architecture"), ["modules"])

        snapshot = board.findings("security")
        snapshot["critical"] = 999
        snapshot["items"].append("mutated")
        self.assertEqual(board.findings("security")["critical"], 3)
        self.assertEqual(board.findings("security")["items"], ["sql", "eval"])

        self.assertEqual(board.require("architecture", "modules"), {"modules": 4})
        with self.assertRaises(ValueError) as ctx:
            board.require("security", "missing_key")
        self.assertIn("missing_key", str(ctx.exception))
        self.assertIn("security", str(ctx.exception))

    def test_unknown_namespace_is_empty(self) -> None:
        board = Blackboard()
        self.assertEqual(board.findings("nope"), {})
        self.assertEqual(board.keys("nope"), [])

    def test_threads_since_cursor_and_stats(self) -> None:
        board = Blackboard()
        board.broadcast("a", "one", thread="t1", kind="assignment")
        board.broadcast("a", "two", thread="t2", kind="summary")
        cursor = board.cursor()
        board.broadcast("b", "three", thread="t1", kind="summary")

        self.assertEqual([m.content for m in board.thread("t1")], ["one", "three"])
        self.assertEqual([m.content for m in board.since(cursor)], ["three"])
        self.assertEqual(board.since(-1), [])
        self.assertEqual(board.since(999), [])

        stats = board.stats()
        self.assertEqual(stats["messages"], 3)
        self.assertEqual(stats["kinds"], {"assignment": 1, "summary": 2})
        self.assertEqual(stats["senders"], {"a": 2, "b": 1})

    def test_dump_is_json_serialisable(self) -> None:
        board = Blackboard()
        board.broadcast("a", "content", kind="summary", score=0.5)
        board.patch("ns", {"k": [1, 2]})
        payload = json.loads(json.dumps(board.dump(), sort_keys=True))
        self.assertEqual(payload["messages"][0]["content"], "content")
        self.assertEqual(payload["findings"]["ns"]["k"], [1, 2])

    def test_clear_and_remove(self) -> None:
        board = Blackboard()
        board.broadcast("a", "x")
        board.patch("ns", {"k": 1})
        board.remove("ns")
        self.assertEqual(board.findings("ns"), {})
        board.clear()
        self.assertEqual(len(board), 0)
        self.assertEqual(board.namespaces(), [])

    def test_concurrent_publish_and_patch(self) -> None:
        """The squad runs waves in threads, so the blackboard must be safe."""
        board = Blackboard()
        errors = []

        def worker(worker_id: int) -> None:
            try:
                for index in range(40):
                    board.broadcast("w%d" % worker_id, "m%d" % index, kind="observation")
                    board.patch("ns-%d" % worker_id, {"k%d" % index: index})
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(board), 8 * 40)
        self.assertEqual(len(board.namespaces()), 8)
        self.assertEqual(len(board.findings("ns-3")), 40)

    def test_stress_helper_reports_no_errors(self) -> None:
        board = Blackboard()
        report = board.stress(writers=4, per_writer=10)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["messages"], 40)


class TestTools(TempWorkspace):
    def setUp(self) -> None:
        super().setUp()
        self.registry = self.registry_default()

    def registry_default(self) -> ToolRegistry:
        self.write("pkg/mod.py", "import os\n\ndef f(x):\n    return eval(x)\n")
        self.write("notes.txt", "hello world\nsecond line\n")
        return self.registry()

    def test_read_file_round_trip(self) -> None:
        result = self.registry.call("read_file", {"path": "pkg/mod.py"})
        self.assertTrue(result.ok, result.error)
        self.assertIn("# pkg/mod.py", result.output)
        self.assertIn("def f(x):", result.output)

    def test_read_file_rejects_escape_and_missing(self) -> None:
        escape = self.registry.call("read_file", {"path": "../outside.txt"})
        self.assertFalse(escape.ok)
        self.assertTrue(escape.error.startswith("PERMISSION") or escape.error.startswith("ERROR"), escape.error)

        absolute = self.registry.call("read_file", {"path": "/etc/hosts"})
        self.assertFalse(absolute.ok)

        missing = self.registry.call("read_file", {"path": "nope.py"})
        self.assertFalse(missing.ok)
        self.assertIn("not a file", missing.error or "")

    def test_absolute_path_inside_workdir_is_allowed(self) -> None:
        result = self.registry.call("read_file", {"path": self.path("notes.txt")})
        self.assertTrue(result.ok, result.error)

    def test_write_file_creates_parents_and_respects_overwrite(self) -> None:
        created = self.registry.call("write_file", {"path": "out/deep/new.txt", "content": "hi"})
        self.assertTrue(created.ok, created.error)
        self.assertIn("wrote out/deep/new.txt", created.output)
        self.assertEqual(self.read("out/deep/new.txt"), "hi")

        refused = self.registry.call(
            "write_file", {"path": "out/deep/new.txt", "content": "x", "overwrite": False}
        )
        self.assertFalse(refused.ok)
        self.assertIn("file exists", refused.error or "")

        escape = self.registry.call("write_file", {"path": "../escape.txt", "content": "x"})
        self.assertFalse(escape.ok)

    def test_list_files_and_search_text(self) -> None:
        listing = self.registry.call("list_files", {"pattern": "**/*.py"})
        self.assertTrue(listing.ok)
        self.assertIn("pkg/mod.py", listing.output)
        self.assertIn("# 1 entries", listing.output)

        found = self.registry.call("search_text", {"query": "eval(", "path": "."})
        self.assertTrue(found.ok)
        self.assertIn("pkg/mod.py:4", found.output)

        none = self.registry.call("search_text", {"query": "no-such-token-xyz"})
        self.assertTrue(none.ok)
        self.assertIn("# 0 matches", none.output)

    def test_python_exec_success_and_failure(self) -> None:
        ok = self.registry.call("python_exec", {"code": "print(6 * 7)"})
        self.assertTrue(ok.ok, ok.error)
        self.assertEqual(ok.output.strip(), "42")

        failed = self.registry.call("python_exec", {"code": "raise SystemExit(3)"})
        self.assertFalse(failed.ok)
        self.assertTrue(failed.error.startswith("EXIT"), failed.error)

        timeout = self.registry.call("python_exec", {"code": "import time; time.sleep(5)", "timeout": 0.4})
        self.assertFalse(timeout.ok)
        self.assertTrue(timeout.error.startswith("TIMEOUT"), timeout.error)

    def test_shell_command_allowlist_and_metacharacters(self) -> None:
        registry = build_default_registry(workdir=self.workdir)
        allowed = registry.call("shell_command", {"command": "echo hello"}, allow_dangerous=True)
        self.assertTrue(allowed.ok, allowed.error)
        self.assertIn("hello", allowed.output)

        denied = registry.call("shell_command", {"command": "rm -rf /"}, allow_dangerous=True)
        self.assertFalse(denied.ok)
        self.assertTrue(denied.error.startswith("PERMISSION"), denied.error)

        chained = registry.call("shell_command", {"command": "echo hi && rm -rf ."}, allow_dangerous=True)
        self.assertFalse(chained.ok)
        self.assertTrue(chained.error.startswith("PERMISSION"), chained.error)

    def test_dangerous_gate(self) -> None:
        """An opt-out registry must refuse dangerous tools; the shipped default
        registry may call them (that is the policy its own agents rely on)."""
        spec = ToolSpec(
            name="python_exec",
            description="runs python",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            dangerous=True,
        )

        def run(code: str = "") -> int:
            return 1

        strict = ToolRegistry()
        strict.register(spec, run)
        self.assertFalse(strict.allow_dangerous)

        blocked = strict.call("python_exec", {"code": "print(1)"})
        self.assertFalse(blocked.ok)
        self.assertTrue(blocked.error.startswith("DANGEROUS"), blocked.error)

        per_call = strict.call("python_exec", {"code": "print(1)"}, allow_dangerous=True)
        self.assertTrue(per_call.ok, per_call.error)

        # The shipped default registry is the framework's own policy: its agents
        # call dangerous tools without an extra flag.
        self.assertTrue(self.registry.allow_dangerous)
        self.assertTrue(self.registry.call("python_exec", {"code": "print(1)"}).ok)

    def test_unknown_and_bad_arguments_never_raise(self) -> None:
        unknown = self.registry.call("does_not_exist", {"a": 1})
        self.assertFalse(unknown.ok)
        self.assertTrue(unknown.error.startswith("NOT_FOUND"), unknown.error)
        self.assertEqual(unknown.name, "does_not_exist")
        self.assertEqual(unknown.arguments, {"a": 1})

        missing = self.registry.call("read_file", {})
        self.assertFalse(missing.ok)
        self.assertTrue(missing.error.startswith("MISSING_ARGS"), missing.error)

        extra = self.registry.call("read_file", {"path": "notes.txt", "bogus": 1})
        self.assertFalse(extra.ok)
        self.assertTrue(extra.error.startswith("EXTRA_ARGS"), extra.error)

        bad_type = self.registry.call("read_file", {"path": "notes.txt", "max_bytes": "abc"})
        self.assertFalse(bad_type.ok)
        self.assertTrue(bad_type.error.startswith("BAD_TYPE"), bad_type.error)

    def test_integer_coercion_from_strings(self) -> None:
        result = self.registry.call("read_file", {"path": "notes.txt", "max_bytes": "10"})
        self.assertTrue(result.ok, result.error)
        self.assertIn("truncated", result.output)

    def test_output_truncation_and_audit_log(self) -> None:
        big = self.registry.call("python_exec", {"code": "print('x' * 5000)", "max_output": 100}, allow_dangerous=True)
        self.assertTrue(big.ok)
        self.assertIn("truncated", big.output)
        self.assertTrue(len(self.registry.audit_log()) >= 1)

    def test_describe_and_describe_json(self) -> None:
        text = self.registry.describe()
        self.assertIn("### read_file", text)
        self.assertIn("### python_exec", text)
        parsed = json.loads(self.registry.describe_json())
        self.assertEqual(parsed[0]["name"], "read_file")

        self.assertEqual(ToolRegistry().describe(), "(no tools available)")
        self.assertEqual(ToolRegistry().call("anything", {}).ok, False)

    def test_call_many_accepts_toolcalls_and_dicts(self) -> None:
        calls = [
            ToolCall(name="read_file", arguments={"path": "notes.txt"}),
            {"name": "list_files", "arguments": {"pattern": "**/*.txt"}},
        ]
        results = self.registry.call_many(calls)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.ok for r in results), [r.error for r in results])

    def test_sandbox_context_manager(self) -> None:
        with self.registry.sandbox(timeout=1.0, max_output=20) as sandbox:
            self.assertIsNot(sandbox, self.registry)
            limited = sandbox.call("list_files", {"pattern": "**/*"})
            self.assertTrue(limited.ok)
            self.assertIn("truncated", limited.output)
            self.assertEqual(sandbox.sandbox_limits["max_output"], 20)
            self.assertTrue(sandbox.audit_log())

    def test_builtin_specs_are_registered(self) -> None:
        names = [spec.name for spec in self.registry.specs()]
        self.assertEqual(
            names,
            ["read_file", "write_file", "list_files", "search_text", "python_exec", "shell_command", "http_get"],
        )
        dangerous = {spec.name for spec in self.registry.specs() if spec.dangerous}
        self.assertTrue({"write_file", "shell_command", "python_exec"} <= dangerous)
        self.assertTrue(self.registry.has("read_file"))
        self.assertFalse(self.registry.has("nope"))

    def test_custom_tool_and_lookup_error(self) -> None:
        registry = ToolRegistry()

        @registry.tool("double", "doubles a number", {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})
        def double(n: int = 0) -> int:
            return n * 2

        self.assertTrue(registry.has("double"))
        self.assertEqual(registry.call("double", {"n": 21}).output, "42")
        spec = registry.specs()[0]
        self.assertIsInstance(spec, ToolSpec)
        self.assertEqual(spec.name, "double")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
