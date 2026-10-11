"""Enterprise Multi-Agent Collaboration Platform - FastAPI backend server.

Provides:
- Real-time WebSocket events for squad execution progress
- REST API endpoints for DAG state queries and human-in-the-loop gating
- Blackboard diff inspector data stream
- Token usage tracking and metrics collection
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Pydantic models for API communication
class ExecutionEvent(BaseModel):
    """Structured event for squad execution progress."""
    type: str  # 'group_start', 'agent_start', 'agent_result', 'blackboard_update', 'review_complete'
    wave: int
    group: Optional[str] = None
    agent: Optional[str] = None
    status: Optional[str] = None  # 'idle', 'thinking', 'executing', 'done', 'failed'
    score: Optional[float] = None
    scores: Optional[Dict[str, float]] = None
    elapsed_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    output: Optional[str] = None
    findings: Optional[Dict[str, Any]] = None


class HILGatingRequest(BaseModel):
    """Human-in-the-loop gating decision."""
    decision: str  # 'approve', 'reject_with_feedback'
    feedback: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class MetricsCollector:
    """Collect and aggregate execution metrics in real-time."""
    start_time: float = field(default_factory=time.time)
    total_tokens: int = 0
    agents_completed: int = 0
    agent_scores: List[float] = field(default_factory=list)
    agent_durations: List[float] = field(default_factory=list)

    def record_agent_completion(self, score: float, elapsed_ms: float, tokens: int = 0):
        self.agent_scores.append(score)
        self.agent_durations.append(elapsed_ms)
        if tokens:
            self.total_tokens += tokens
        self.agents_completed += 1

    @property
    def avg_score(self) -> float:
        return sum(self.agent_scores) / len(self.agent_scores) if self.agent_scores else 0

    @property
    def total_duration(self) -> float:
        return sum(self.agent_durations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_agents': self.agents_completed,
            'avg_score': round(self.avg_score, 3),
            'total_tokens': self.total_tokens,
            'total_duration_ms': round(self.total_duration, 0),
            'duration_seconds': round(self.total_duration / 1000, 2)
        }


@dataclass
class BlackboardDiffTracker:
    """Track state changes on the blackboard for diff visualization."""
    snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def capture_snapshot(self, namespace: str, findings: Dict[str, Any]):
        self.snapshots[namespace] = {
            'timestamp': time.time(),
            'findings': findings
        }
    
    def get_diff(self, namespace: str) -> Optional[Dict[str, Any]]:
        return self.snapshots.get(namespace)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.snapshots


class SquadExecutionMonitor:
    """Monitor squad execution and emit events to all connected WebSocket clients."""
    
    def __init__(self):
        self.connected_clients: Set[WebSocket] = set()
        self.metrics = MetricsCollector()
        self.blackboard_diffs = BlackboardDiffTracker()
        self.current_execution: Optional[Dict[str, Any]] = None
    
    async def connect_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.connected_clients.add(websocket)
        
        # Send initial metrics snapshot
        await websocket.send_json({
            'type': 'metrics_update',
            'metrics': self.metrics.to_dict()
        })
    
    def disconnect_websocket(self, websocket: WebSocket):
        self.connected_clients.discard(websocket)
    
    def broadcast_event(self, event: ExecutionEvent):
        """Broadcast event to all connected WebSocket clients."""
        if not self.connected_clients:
            return
        
        loop = asyncio.get_event_loop()
        event_data = event.model_dump(exclude_none=True)
        
        for client in list(self.connected_clients):
            try:
                loop.run_in_executor(None, lambda e=event_data, w=client: asyncio.create_task(
                    w.send_json(e)
                ))
            except Exception:
                pass  # Client disconnected silently
    
    def update_metrics(self, score: float, elapsed_ms: float, tokens: int = 0):
        """Update internal metrics with agent completion data."""
        self.metrics.record_agent_completion(score, elapsed_ms, tokens)
        
        # Broadcast updated metrics
        self.broadcast_event(ExecutionEvent(
            type='metrics_update',
            wave=0,
            metrics=self.metrics.to_dict()
        ))
    
    def track_blackboard_change(self, namespace: str, findings: Dict[str, Any]):
        """Track a change to the blackboard for diff visualization."""
        self.blackboard_diffs.capture_snapshot(namespace, findings)
        
        self.broadcast_event(ExecutionEvent(
            type='blackboard_update',
            wave=0,
            agent=namespace,
            findings=findings
        ))
    
    async def handle_hil_decision(self, request: HILGatingRequest, callback: Callable):
        """Handle human-in-the-loop gating decision."""
        decision = request.decision
        
        if decision == 'approve':
            message = "✅ 审批通过，智能体继续执行后续任务"
            await self.broadcast_event(ExecutionEvent(
                type='hil_decision',
                status='approved',
                message=message
            ))
        elif decision == 'reject_with_feedback':
            message = f"❌ 已打回 - 反馈：{request.feedback}"
            await self.broadcast_event(ExecutionEvent(
                type='hil_rejection',
                status='rejected',
                feedback=request.feedback,
                message=message
            ))
            
            # Trigger re-execution via callback
            await callback(request.feedback)
    
    async def notify_review_complete(self, result: Dict[str, Any]):
        """Notify all clients that review phase is complete."""
        self.broadcast_event(ExecutionEvent(
            type='review_complete',
            wave=4,
            result=result
        ))


# Global monitor instance
monitor = SquadExecutionMonitor()

# Create FastAPI app
app = FastAPI(
    title="Enterprise Multi-Agent Collaboration Platform",
    description="Real-time DAG visualization and HIL gating for multi-agent squads",
    version="2.0.0"
)

# Mount static files (frontend HTML)
app.mount("/static", StaticFiles(directory="/Users/albert/orca/workspaces/bisai/11-genai-open-agent/11_genai-open-agent/frontend"), name="frontend")


@app.get("/")
async def root():
    """Serve the main dashboard."""
    with open("/Users/albert/orca/workspaces/bisai/11-genai-open-agent/11_genai-open-agent/frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time squad execution events."""
    await monitor.connect_websocket(websocket)
    try:
        while True:
            # Listen for HIL gating decisions
            data = await websocket.receive_text()
            try:
                request = HILGatingRequest(**json.loads(data))
                await monitor.handle_hil_decision(request, _retrigger_agent)
            except Exception:
                pass  # Ignore malformed requests
    except WebSocketDisconnect:
        monitor.disconnect_websocket(websocket)


async def _retrigger_agent(feedback: str):
    """Callback to trigger agent re-execution after rejection."""
    # In production, this would interface with the squad orchestration
    # to restart the failed agent with the feedback incorporated
    print(f"Re-triggering failed agent with feedback: {feedback}")


@app.get("/metrics")
async def get_metrics():
    """Get current execution metrics."""
    return {"metrics": monitor.metrics.to_dict()}


@app.get("/blackboard-diffs")
async def get_blackboard_diffs():
    """Get blackboard diff history for visualization."""
    return {"diffs": monitor.blackboard_diffs.to_dict()}


@app.post("/hil/gating")
async def hil_gating(request: HILGatingRequest):
    """Submit human-in-the-loop gating decision."""
    await monitor.handle_hil_decision(request, _retrigger_agent)
    return {"status": "processed", "decision": request.decision}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server using uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
