# Demo video script (2–5 minutes)

Target: 3:30. Everything shown is reproducible by a judge with one command.
Record at 1440p, terminal font ≥ 16pt, hide unrelated windows.

## 0:00 — 0:20 · Hook and problem

**Screen:** this README, scrolled to the three bullets.

> "Most agent demos are one prompt, one answer. This is an agent *squad*: four
> specialists that coordinate through a shared blackboard, critique their own
> output against a rubric, and prove their claims by executing sandboxed tools.
> Pure Python standard library — no API key needed to see it work."

## 0:20 — 0:50 · One-command proof

**Terminal:**

```bash
docker run --rm agent-squad        # or: PYTHONPATH=src python3 -m squad.cli demo
```

Call out the live event stream: `wave 1 -> group architecture … group review done`
and the final `status: PASS` with per-agent scores.

**Voice:** "No credentials, no network. The same code path runs against a real
OpenAI-compatible endpoint when `OMP_API_KEY` is set — only the provider is
swapped."

## 0:50 — 1:40 · Multi-agent collaboration

**Screen:** `docs/ARCHITECTURE.md` §3 diagram + `squad plan` output.

> "Agents are grouped: architecture → security → tests → review. `plan()` is a
> topological sort with sorted tie-breaking, so the wave list is identical on
> every run. Waves are separated by a barrier because a reproducible run cannot
> depend on a sibling's in-flight state. Downstream agents receive upstream
> findings by namespace — that is the whole inter-agent protocol."

Show the blackboard dump: `BLACKBOARD / messages: 8 / namespaces: audit-architecture, …`.

## 1:40 — 2:30 · Self-reflection loop

**Screen:** `src/squad/agent.py` — the reflection round in `execute()`, then
`docs/ARCHITECTURE.md` §4 table.

> "Each round: one model call, tools executed, then a deterministic five-dimension
> score — structure, coverage, evidence, specificity, actionability. Below 0.75,
> the agent is rejected and the critique names exactly what was missing, in
> ascending score order, plus a concrete instruction. That critique becomes the
> next round's prompt."

**Then show a deliberate failure:** run an agent with `--max-rounds 1` on a vague
task and point at `status: unfinished` + the critique line + `stagnation at round 1`
in the warnings.

> "Reflection is not decoration: you can watch the loop reject its own first draft."

## 2:30 — 3:15 · Closed-loop tool execution

**Screen:** `src/squad/tools.py` — the validation order in `call()`, then a live
`python -m squad.cli audit ./src --verbose` run showing `tool calls : read_file,
read_file, list_files, search_text, python_exec`.

> "Tools are the only evidence channel. Every call is validated before it runs —
> unknown tool, missing or extra arguments, hallucinated types, workdir escapes,
> shell metacharacters, timeouts, output caps. Nothing raises; everything returns
> a coded result that is fed back as an observation. The Test Engineer agent even
> writes and runs a real regression test."

**Screen:** run the escape attempt live:

```python
python -c "from squad.tools import build_default_registry as b; print(b(workdir='/tmp/x').call('read_file', {'path': '/etc/hosts'}).error)"
```

## 3:15 — 3:30 · Close

**Screen:** `tests/` file list, suite summary line, repo footer.

> "65 tests: blackboard under 8-thread contention, tool sandbox escapes, rubric
> determinism, provider fallback, orchestration isolation, and the full CLI
> end-to-end. Reproducible, dependency-free, and it runs in one command."

## Recording checklist

- [ ] `docker build -t agent-squad .` succeeds on a clean machine
- [ ] `docker run --rm agent-squad` prints `status: PASS`
- [ ] `PYTHONPATH=src python3 -m unittest discover -s tests -t .` is green
- [ ] no API key visible on screen (`env | grep -i key` scrubbed)
- [ ] video length 2–5 min, English narration, upload to YouTube/Loom (unlisted is fine)
