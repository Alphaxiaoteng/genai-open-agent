"""tests/test_checkpoint.py - 状态机快照时光机与 Token 熔断器测试套件。"""

import pytest
from squad.checkpoint import (
    AgentTopologyRouter,
    BudgetExceededError,
    CheckpointManager,
    CircuitState,
    TokenBudgetCircuitBreaker,
)


def test_checkpoint_record_and_rollback():
    mgr = CheckpointManager(max_snapshots=10)

    mgr.record_checkpoint("step_1", "researcher", {"finding": "alpha", "score": 10})
    mgr.record_checkpoint("step_2", "writer", {"draft": "hello world", "version": 1})
    mgr.record_checkpoint("step_3", "reviewer", {"feedback": "needs polish", "pass": False})

    assert len(mgr.history()) == 3
    latest = mgr.get_latest_state()
    assert latest["pass"] is False

    # 回滚到 step_2
    restored = mgr.rollback_to("step_2")
    assert restored["version"] == 1
    assert "feedback" not in restored
    assert len(mgr.history()) == 2


def test_checkpoint_fork_branch():
    mgr = CheckpointManager(max_snapshots=10)

    mgr.record_checkpoint("s1", "n1", {"val": 1})
    mgr.record_checkpoint("s2", "n2", {"val": 2})

    # 从 s1 分叉
    branched_state = mgr.rollback_to("s1", fork_new_branch=True, new_branch_name="alternate_path")
    assert branched_state["val"] == 1

    # 在新分支记录
    mgr.record_checkpoint("s1_b", "n1_alt", {"val": 99})
    assert len(mgr.history()) == 2
    assert mgr.get_latest_state()["val"] == 99


def test_token_budget_circuit_breaker():
    breaker = TokenBudgetCircuitBreaker(
        max_total_tokens=10_000,
        max_turn_tokens=6_000,
        warning_ratio=0.8,
        cost_per_1k_cny=0.015,
    )

    # 正常调用
    s1 = breaker.record_usage("coder", prompt_tokens=2_000, completion_tokens=1_000)
    assert s1 == CircuitState.NORMAL
    assert breaker.total_usage.total_tokens == 3_000

    # 接近告警线
    s2 = breaker.record_usage("reviewer", prompt_tokens=3_500, completion_tokens=2_000)
    assert s2 == CircuitState.WARNING
    assert breaker.total_usage.total_tokens == 8_500
    assert breaker.estimated_cost_cny() > 0

    # 单轮超限异常 (7000 > 6000)
    with pytest.raises(BudgetExceededError, match="单轮安全限额"):
        breaker.record_usage("coder", prompt_tokens=4_000, completion_tokens=3_000)

    # 总限额超限熔断 (8500 + 2000 = 10500 > 10000)
    with pytest.raises(BudgetExceededError, match="触发熔断阈值"):
        breaker.record_usage("coder", prompt_tokens=1_000, completion_tokens=1_000)


def test_agent_topology_router_dag_order():
    router = AgentTopologyRouter()

    # 数据挖掘 -> 分析器 & 架构师 -> 报告汇总
    router.add_agent_node("ingestion")
    router.add_agent_node("analyzer", prerequisites=["ingestion"])
    router.add_agent_node("architect", prerequisites=["ingestion"])
    router.add_agent_node("reporter", prerequisites=["analyzer", "architect"])

    assert router.validate_dag() is True

    layers = router.get_execution_order()
    assert layers == [
        ["ingestion"],
        ["analyzer", "architect"],
        ["reporter"],
    ]


def test_agent_topology_router_cycle_detection():
    router = AgentTopologyRouter()
    # A -> B -> C -> A (环)
    router.add_agent_node("A", prerequisites=["C"])
    router.add_agent_node("B", prerequisites=["A"])
    router.add_agent_node("C", prerequisites=["B"])

    assert router.validate_dag() is False
    with pytest.raises(ValueError, match="循环依赖"):
        router.get_execution_order()
