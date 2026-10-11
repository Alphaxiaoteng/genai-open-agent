#!/usr/bin/env python3
"""CLI for running squad with web dashboard integration.

This module extends the existing squad orchestration with:
- Real-time event emission via WebSocket
- Token usage tracking and metrics aggregation
- Human-in-the-loop gating hooks
- Blackboard diff capture for visualization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from squad.cli import main as cli_main
from squad.memory import Blackboard
from squad.squad import Squad, SquadResult
from squads.code_audit import build_code_audit_squad


class WebEventEmitter:
    """Emits real-time events during squad execution for WebSocket clients."""
    
    def __init__(self):
        self.event_log: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    def emit(self, event_type: str, **data):
        """Record an event that will be broadcast via WebSocket."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            **data
        }
        self.event_log.append(event)
        
        # In production, this would broadcast via monitor.broadcast_event()
        print(f"[EVENT] {event_type}: {json.dumps(data, ensure_ascii=False)}")
    
    def on_group_start(self, wave: int, group: str):
        self.emit('group_start', wave=wave, group=group, status='started')
    
    def on_agent_start(self, agent: str, wave: int, group: str):
        self.emit('agent_start', wave=wave, group=group, agent=agent, status='executing')
    
    def on_agent_result(self, result: 'AgentResult', wave: int, group: str):
        self.emit('agent_result', 
                 wave=wave, 
                 group=group, 
                 agent=result.agent,
                 status=result.status,
                 score=result.score,
                 scores=result.scores,
                 elapsed_ms=result.elapsed_ms,
                 output=result.output[:300] if result.output else None)
        
        # Update metrics
        current_metrics = self.metrics.get('metrics', {})
        current_metrics['agents_completed'] = current_metrics.get('agents_completed', 0) + 1
        current_metrics['agent_scores'] = current_metrics.get('agent_scores', []) + [result.score]
        current_metrics['agent_durations'] = current_metrics.get('agent_durations', []) + [result.elapsed_ms]
        
        scores = current_metrics['agent_scores']
        current_metrics['avg_score'] = round(sum(scores) / len(scores), 3) if scores else 0
        
        self.metrics['metrics'] = current_metrics
    
    def on_blackboard_update(self, namespace: str, findings: Dict[str, Any]):
        self.emit('blackboard_update', namespace=namespace, findings=findings)


def squad_event_handler(event: Dict[str, Any], emitter: WebEventEmitter):
    """Convert squad events to WebEventEmitter calls."""
    event_type = event.get('type')
    
    if event_type == 'group' and event.get('status') == 'started':
        emitter.on_group_start(wave=event['wave'], group=event['group'])
    
    elif event_type == 'agent' and event.get('status') == 'started':
        emitter.on_agent_start(agent=event['agent'], wave=event['wave'], group=event.get('group'))
    
    elif event_type == 'result':
        # We need AgentResult object here
        pass  # Will be handled differently in run_audit


def collect_source_files(root: str, *, include: tuple = ('py', 'md')) -> list:
    """Return auditable files as relative POSIX paths, sorted."""
    skip_dirs = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.pytest_cache', '.agent-squad'}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for name in sorted(filenames):
            ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ""
            if ext not in include:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            out.append(rel.replace(os.sep, '/'))
    return sorted(out)


def run_audit_with_web(
    paths: List[str],
    squad: Squad,
    blackboard: Optional[Blackboard] = None,
    focus: str = "",
    on_event: Optional[Callable] = None,
) -> SquadResult:
    """Run squad with enhanced event handling for web dashboard."""
    from squad.workflow import EventPrinter, build_task_text
    
    board = blackboard if blackboard is not None else squad.blackboard
    emitter = WebEventEmitter()
    
    for agent in squad:
        objective = squad.objectives().get(agent.name, agent.goal or agent.name)
        agent.prompt = build_task_text(paths, focus=focus, objective=objective)
        agent.task = agent.prompt
    
    # Create custom event handler that captures results
    def wrapped_on_event(event: Dict[str, Any]):
        # Call original handler first
        if on_event:
            on_event(event)
        
        # Then emit to web dashboard
        event_type = event.get('type')
        
        if event_type == 'group' and event.get('status') == 'started':
            emitter.on_group_start(wave=event['wave'], group=event['group'])
        elif event_type == 'agent' and event.get('status') == 'started':
            emitter.on_agent_start(agent=event['agent'], wave=event['wave'], group=event.get('group'))
        elif event_type == 'result':
            emitter.on_agent_result(
                event.get('result'),
                wave=event.get('wave'),
                group=event.get('group')
            )
    
    return squad.run(focus=focus, blackboard=board, on_event=wrapped_on_event)


def render_dashboard_report(result: SquadResult, metrics: Dict[str, Any], blackboard: Optional[Blackboard] = None) -> str:
    """Render enhanced report with metrics."""
    lines = []
    lines.append("=" * 72)
    lines.append("ENTERPRISE MULTI-AGENT COLLABORATION PLATFORM - EXECUTION REPORT")
    lines.append("=" * 72)
    lines.append("squad      : %s" % result.name)
    lines.append("status     : %s" % ("PASS" if result.ok() else "NEEDS WORK"))
    lines.append("")
    
    # Add metrics section
    m = metrics.get('metrics', {})
    lines.append("EXECUTION METRICS")
    lines.append("-" * 72)
    lines.append("Total Agents Completed: %d" % m.get('agents_completed', 0))
    lines.append("Average Score: %.3f" % m.get('avg_score', 0))
    lines.append("Duration (ms): %.0f" % m.get('total_duration', 0))
    lines.append("")
    
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Multi-Agent Collaboration Platform - Web Dashboard Mode")
    parser.add_argument("--web", action="store_true", help="Enable web dashboard mode")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--workspace", default=None, help="Audit this directory instead of generated sample")
    parser.add_argument("--offline", action="store_true", help="Use MockLLM for deterministic execution")
    
    args = parser.parse_args(argv)
    
    if args.web:
        # Import web server after parsing args
        from squad.web_server import run_server as start_web_server
        
        # Start web server in background thread
        import threading
        server_thread = threading.Thread(target=start_web_server, args=(args.host, args.port), daemon=True)
        server_thread.start()
        
        print(f"🚀 Enterprise Multi-Agent Platform started at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server\n")
        
        # Continue with regular CLI execution
        sys.argv.pop(sys.argv.index('--web'))
        sys.argv.pop(sys.argv.index(args.host))
        sys.argv.pop(sys.argv.index(str(args.port)))
    
    # Run the main CLI (which will use the existing demo/audit logic)
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
