"""checkpoint.py - Agent DAG 状态机快照、时间旅行回滚与 Token 预算熔断器。

为大厂生产级多 Agent 编排提供：
1. 确定性状态机快照与时光机回滚 (Time-travel Rollback)。
2. 细粒度 Token 预算熔断器 (TokenBudgetCircuitBreaker)，防止长任务死循环导致的费用爆炸。
3. 拓扑感知多 Agent 任务路由器 (AgentTopologyRouter)，支持 DAG 依赖解析与环检测。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class CircuitState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    TRIPPED = "tripped"


class BudgetExceededError(Exception):
    """Token 预算耗尽熔断异常。"""


@dataclass
class Snapshot:
    step_id: str
    step_index: int
    node_name: str
    state_payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    parent_step_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Agent DAG 执行状态机快照管理器与时光机回滚器。"""

    def __init__(self, max_snapshots: int = 100) -> None:
        self.max_snapshots = max_snapshots
        self._timeline: List[Snapshot] = []
        self._branches: Dict[str, List[Snapshot]] = {"main": self._timeline}
        self._current_branch: str = "main"

    def record_checkpoint(
        self,
        step_id: str,
        node_name: str,
        state: Dict[str, Any],
        parent_step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """记录一个不可变状态快照。"""
        # 深拷贝隔离，防止调用方修改破坏快照确定性
        cloned_state = copy.deepcopy(state)
        step_index = len(self._timeline)
        snap = Snapshot(
            step_id=step_id,
            step_index=step_index,
            node_name=node_name,
            state_payload=cloned_state,
            parent_step_id=parent_step_id,
            metadata=metadata or {},
        )
        self._timeline.append(snap)
        if len(self._timeline) > self.max_snapshots:
            self._timeline.pop(0)
        return snap

    def rollback_to(self, step_id: str, fork_new_branch: bool = False, new_branch_name: Optional[str] = None) -> Dict[str, Any]:
        """时光机回滚：恢复到指定 step_id 的状态。
        
        若 fork_new_branch=True，则从该点分叉出新分支，保留旧历史；
        否则截断当前分支后续步骤。
        """
        target_idx = -1
        for i, snap in enumerate(self._timeline):
            if snap.step_id == step_id:
                target_idx = i
                break
        
        if target_idx == -1:
            raise KeyError(f"未找到指定的快照标识: {step_id}")

        target_snap = self._timeline[target_idx]

        if fork_new_branch:
            branch_name = new_branch_name or f"branch_{step_id}_{int(time.time())}"
            # 克隆截至目标快照的历史
            new_timeline = copy.deepcopy(self._timeline[: target_idx + 1])
            self._branches[branch_name] = new_timeline
            self._current_branch = branch_name
            self._timeline = new_timeline
        else:
            # 截断回滚
            self._timeline = self._timeline[: target_idx + 1]

        return copy.deepcopy(target_snap.state_payload)

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        if not self._timeline:
            return None
        return copy.deepcopy(self._timeline[-1].state_payload)

    def history(self) -> List[Dict[str, Any]]:
        return [
            {
                "step_id": s.step_id,
                "step_index": s.step_index,
                "node_name": s.node_name,
                "timestamp": s.timestamp,
                "parent_step_id": s.parent_step_id,
            }
            for s in self._timeline
        ]


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenBudgetCircuitBreaker:
    """企业级 Token 预算熔断器。"""

    def __init__(
        self,
        max_total_tokens: int = 100_000,
        max_turn_tokens: int = 8_000,
        warning_ratio: float = 0.8,
        cost_per_1k_cny: float = 0.012,  # 综合模型估算单价
    ) -> None:
        self.max_total_tokens = max_total_tokens
        self.max_turn_tokens = max_turn_tokens
        self.warning_ratio = warning_ratio
        self.cost_per_1k_cny = cost_per_1k_cny

        self.usage_by_agent: Dict[str, TokenUsage] = {}
        self.total_usage = TokenUsage()
        self.state: CircuitState = CircuitState.NORMAL

    def record_usage(
        self,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> CircuitState:
        """记录单次 Agent 调用消耗，并判定是否触发熔断。"""
        turn_tokens = prompt_tokens + completion_tokens
        if turn_tokens > self.max_turn_tokens:
            self.state = CircuitState.TRIPPED
            raise BudgetExceededError(
                f"单轮调用 Token 消耗 ({turn_tokens}) 超出单轮安全限额 ({self.max_turn_tokens})"
            )

        if agent_name not in self.usage_by_agent:
            self.usage_by_agent[agent_name] = TokenUsage()

        agent_rec = self.usage_by_agent[agent_name]
        agent_rec.prompt_tokens += prompt_tokens
        agent_rec.completion_tokens += completion_tokens
        agent_rec.cached_tokens += cached_tokens
        agent_rec.total_calls += 1

        self.total_usage.prompt_tokens += prompt_tokens
        self.total_usage.completion_tokens += completion_tokens
        self.total_usage.cached_tokens += cached_tokens
        self.total_usage.total_calls += 1

        if self.total_usage.total_tokens >= self.max_total_tokens:
            self.state = CircuitState.TRIPPED
            raise BudgetExceededError(
                f"总 Token 消耗 ({self.total_usage.total_tokens}) 触发熔断阈值 ({self.max_total_tokens})"
            )
        elif self.total_usage.total_tokens >= self.max_total_tokens * self.warning_ratio:
            self.state = CircuitState.WARNING
        else:
            self.state = CircuitState.NORMAL

        return self.state

    def estimated_cost_cny(self) -> float:
        """估算累计产生的费用 (元)。"""
        return round((self.total_usage.total_tokens / 1000.0) * self.cost_per_1k_cny, 4)

    def summary(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "total_tokens": self.total_usage.total_tokens,
            "prompt_tokens": self.total_usage.prompt_tokens,
            "completion_tokens": self.total_usage.completion_tokens,
            "cached_tokens": self.total_usage.cached_tokens,
            "total_calls": self.total_usage.total_calls,
            "estimated_cost_cny": self.estimated_cost_cny(),
            "remaining_tokens": max(0, self.max_total_tokens - self.total_usage.total_tokens),
            "by_agent": {
                name: {
                    "total_tokens": u.total_tokens,
                    "calls": u.total_calls,
                }
                for name, u in self.usage_by_agent.items()
            },
        }

    def reset(self) -> None:
        """重置账本与熔断器。"""
        self.usage_by_agent.clear()
        self.total_usage = TokenUsage()
        self.state = CircuitState.NORMAL


class AgentTopologyRouter:
    """多 Agent DAG 拓扑分析与依赖调度器。"""

    def __init__(self) -> None:
        self._nodes: Set[str] = set()
        self._dependencies: Dict[str, Set[str]] = {}  # node -> set of prerequisites

    def add_agent_node(self, agent_name: str, prerequisites: Optional[List[str]] = None) -> None:
        self._nodes.add(agent_name)
        if agent_name not in self._dependencies:
            self._dependencies[agent_name] = set()
        if prerequisites:
            for pre in prerequisites:
                self._nodes.add(pre)
                self._dependencies[agent_name].add(pre)

    def validate_dag(self) -> bool:
        """检查是否有循环依赖。有环返回 False。"""
        visited: Dict[str, int] = {}  # 0: 未访问, 1: 访问中, 2: 已完成

        def dfs(node: str) -> bool:
            visited[node] = 1
            for pre in self._dependencies.get(node, set()):
                if visited.get(pre, 0) == 1:
                    return False  # 发现环
                if visited.get(pre, 0) == 0:
                    if not dfs(pre):
                        return False
            visited[node] = 2
            return True

        for node in self._nodes:
            if visited.get(node, 0) == 0:
                if not dfs(node):
                    return False
        return True

    def get_execution_order(self) -> List[List[str]]:
        """获取分层并行拓扑执行计划 (Topological Layers)。"""
        if not self.validate_dag():
            raise ValueError("DAG 存在循环依赖，无法生成拓扑执行序")

        in_degree: Dict[str, int] = {node: 0 for node in self._nodes}
        graph: Dict[str, Set[str]] = {node: set() for node in self._nodes}

        for node, prereqs in self._dependencies.items():
            in_degree[node] = len(prereqs)
            for pre in prereqs:
                graph[pre].add(node)

        layers: List[List[str]] = []
        ready = [node for node, deg in in_degree.items() if deg == 0]

        while ready:
            ready.sort()  # 保证确定性
            layers.append(ready)
            next_ready = []
            for node in ready:
                for child in graph[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ready.append(child)
            ready = next_ready

        return layers
