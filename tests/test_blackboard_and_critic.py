"""tests/test_blackboard_and_critic.py - 共享黑板事件总线与自主反思循环测试套件。"""

import pytest
from squad.blackboard import BlackboardEventBus
from squad.critic import CritiqueReport, SelfReflectionCriticLoop


def test_blackboard_pub_sub_and_wildcard():
    bus = BlackboardEventBus()

    bus.register_subscriber("auditor", "task.*")
    bus.register_subscriber("monitor", "*")

    m1 = bus.publish("task.created", sender="leader", payload={"id": "T1"})
    m2 = bus.publish("system.heartbeat", sender="daemon", payload={"ok": True})

    # auditor 仅订阅 task.*
    inbox_auditor = bus.poll_messages("auditor")
    assert len(inbox_auditor) == 1
    assert inbox_auditor[0].topic == "task.created"

    # monitor 订阅 * (收到两条)
    inbox_monitor = bus.poll_messages("monitor")
    assert len(inbox_monitor) == 2

    # 重放事件
    replayed = bus.replay_from(offset=0)
    assert len(replayed) == 2


def test_blackboard_atomic_facts_versioning():
    bus = BlackboardEventBus()

    f1 = bus.set_fact("project_goal", "构建工业级智能体", agent_name="pm")
    assert f1.version == 1
    assert bus.get_fact("project_goal") == "构建工业级智能体"

    f2 = bus.set_fact("project_goal", "构建具备自愈能力的工业级智能体", agent_name="architect")
    assert f2.version == 2
    assert bus.get_fact("project_goal") == "构建具备自愈能力的工业级智能体"


def test_self_reflection_critic_loop_pass_on_second_round():
    loop = SelfReflectionCriticLoop(pass_threshold=80.0, max_rounds=3)

    # 模拟生成器：根据反思反馈补全逻辑
    def generator(prompt: str, feedback: str = None) -> str:
        if not feedback:
            return "简短回答，缺乏深度"
        return "经过批评建议补充完整步骤后的生产级方案"

    # 模拟批评器：第一轮打 60 分，第二轮打 90 分
    call_count = 0

    def critic(prompt: str, draft: str) -> CritiqueReport:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return CritiqueReport(
                iteration=1,
                accuracy_score=60.0,
                completeness_score=50.0,
                conciseness_score=80.0,
                safety_score=90.0,
                critique_notes=["论证不完整", "缺少具体步骤"],
                suggested_revision="请补充生产级完整步骤",
            )
        else:
            return CritiqueReport(
                iteration=2,
                accuracy_score=95.0,
                completeness_score=90.0,
                conciseness_score=85.0,
                safety_score=95.0,
                critique_notes=[],
                suggested_revision="",
            )

    res = loop.execute_loop("请给出调度系统方案", generator, critic)
    assert res.is_passed is True
    assert res.iterations_run == 2
    assert res.final_score > 85.0
    assert "生产级" in res.final_output
    assert "完整步骤" in res.final_output
