from __future__ import annotations

import pytest
from squad.raft import (
    NodeRole,
    RaftConsensusCluster,
    RaftNode,
)


def test_raft_cluster_election_quorum():
    cluster = RaftConsensusCluster(node_ids=["agent_planner", "agent_coder", "agent_reviewer"])
    assert cluster.leader_id is None

    # agent_planner 参选，获得多数派支持（3 节点中获得至少 2 票）
    won = cluster.elect_leader("agent_planner")
    assert won is True
    assert cluster.leader_id == "agent_planner"
    assert cluster.nodes["agent_planner"].role == NodeRole.LEADER
    assert cluster.nodes["agent_coder"].role == NodeRole.FOLLOWER
    assert cluster.nodes["agent_reviewer"].role == NodeRole.FOLLOWER


def test_raft_cluster_log_replication_and_commit():
    cluster = RaftConsensusCluster(node_ids=["agent_1", "agent_2", "agent_3"])
    cluster.elect_leader("agent_1")

    # Leader 提交指令
    task_cmd = {"action": "EXECUTE_RESEARCH", "topic": "autonomous_consensus"}
    success = cluster.propose(task_cmd)
    assert success is True

    # 验证 Leader 与 Followers 均已提交并应用日志
    leader = cluster.nodes["agent_1"]
    assert leader.commit_index == 1
    assert len(leader.applied_commands) == 1
    assert leader.applied_commands[0]["action"] == "EXECUTE_RESEARCH"

    follower = cluster.nodes["agent_2"]
    assert follower.commit_index == 1
    assert len(follower.applied_commands) == 1
    assert follower.applied_commands[0]["topic"] == "autonomous_consensus"
