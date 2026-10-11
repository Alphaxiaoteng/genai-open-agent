"""squad.raft - 多智能体集群分布式 Raft 一致性状态机。

提供严谨的 Raft 算法核心：
1. 节点三态机 (Follower, Candidate, Leader) 与选举任期 (Term)。
2. RequestVote RPC: 候选者拉票、法定人数判定 (Majority Quorum >= N//2 + 1) 与选票自投。
3. AppendEntries RPC: 领导者心跳租约、日志强一致复制与 Commit Index 提交状态机。
4. 发现更高任期强制降级 (Step Down) 与日志安全性保障。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NodeRole(str, Enum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"


@dataclass
class LogEntry:
    term: int
    index: int
    command: Dict[str, Any]


@dataclass
class RequestVoteArgs:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteReply:
    term: int
    vote_granted: bool


@dataclass
class AppendEntriesArgs:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesReply:
    term: int
    success: bool
    match_index: int


class RaftNode:
    """单个智能体 Raft 一致性节点。"""

    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = list(peers)

        # 持久化状态
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []

        # 易失性状态
        self.commit_index = 0
        self.last_applied = 0
        self.role = NodeRole.FOLLOWER
        self.applied_commands: List[Dict[str, Any]] = []

    @property
    def last_log_index(self) -> int:
        return len(self.log)

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def start_election(self) -> RequestVoteArgs:
        """从 Follower/Candidate 发起新一轮选举。"""
        self.role = NodeRole.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id

        return RequestVoteArgs(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=self.last_log_index,
            last_log_term=self.last_log_term,
        )

    def handle_request_vote(self, args: RequestVoteArgs) -> RequestVoteReply:
        """处理拉票请求。"""
        if args.term > self.current_term:
            self.current_term = args.term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None

        vote_granted = False
        if args.term == self.current_term and (self.voted_for is None or self.voted_for == args.candidate_id):
            # 检查候选者日志是否至少与自己一样新
            is_up_to_date = (
                args.last_log_term > self.last_log_term
                or (args.last_log_term == self.last_log_term and args.last_log_index >= self.last_log_index)
            )
            if is_up_to_date:
                vote_granted = True
                self.voted_for = args.candidate_id

        return RequestVoteReply(term=self.current_term, vote_granted=vote_granted)

    def handle_append_entries(self, args: AppendEntriesArgs) -> AppendEntriesReply:
        """处理日志追加与心跳。"""
        if args.term > self.current_term:
            self.current_term = args.term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None

        if args.term < self.current_term:
            return AppendEntriesReply(term=self.current_term, success=False, match_index=self.last_log_index)

        # 确立 Leader 合法地位
        self.role = NodeRole.FOLLOWER

        # 检查日志前缀匹配
        if args.prev_log_index > 0:
            if args.prev_log_index > len(self.log):
                return AppendEntriesReply(term=self.current_term, success=False, match_index=self.last_log_index)
            if self.log[args.prev_log_index - 1].term != args.prev_log_term:
                return AppendEntriesReply(term=self.current_term, success=False, match_index=self.last_log_index)

        # 截断冲突日志并追加新条目
        insert_idx = args.prev_log_index
        for entry in args.entries:
            if insert_idx < len(self.log):
                if self.log[insert_idx].term != entry.term:
                    self.log = self.log[:insert_idx]
                    self.log.append(entry)
            else:
                self.log.append(entry)
            insert_idx += 1

        # 更新提交索引
        if args.leader_commit > self.commit_index:
            self.commit_index = min(args.leader_commit, len(self.log))
            self._apply_logs()

        return AppendEntriesReply(term=self.current_term, success=True, match_index=len(self.log))

    def _apply_logs(self) -> None:
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            self.applied_commands.append(entry.command)


class RaftConsensusCluster:
    """管理多智能体集群间 Raft 消息路由与提案共识。"""

    def __init__(self, node_ids: List[str]):
        self.nodes = {nid: RaftNode(nid, [p for p in node_ids if p != nid]) for nid in node_ids}
        self.leader_id: Optional[str] = None

    def elect_leader(self, candidate_id: str) -> bool:
        """发起一轮确定性选举。"""
        cand = self.nodes[candidate_id]
        args = cand.start_election()
        votes = 1  # 自投 1 票

        for peer_id in cand.peers:
            peer = self.nodes[peer_id]
            reply = peer.handle_request_vote(args)
            if reply.vote_granted:
                votes += 1

        quorum = (len(self.nodes) // 2) + 1
        if votes >= quorum:
            cand.role = NodeRole.LEADER
            self.leader_id = candidate_id
            return True
        return False

    def propose(self, command: Dict[str, Any]) -> bool:
        """通过 Leader 提交指令并同步到多数派节点。"""
        if not self.leader_id or self.nodes[self.leader_id].role != NodeRole.LEADER:
            return False

        leader = self.nodes[self.leader_id]
        new_index = leader.last_log_index + 1
        new_entry = LogEntry(term=leader.current_term, index=new_index, command=command)
        leader.log.append(new_entry)

        # 广播 AppendEntries
        ack_count = 1  # Leader 自身
        for peer_id in leader.peers:
            peer = self.nodes[peer_id]
            args = AppendEntriesArgs(
                term=leader.current_term,
                leader_id=leader.node_id,
                prev_log_index=new_index - 1,
                prev_log_term=leader.log[new_index - 2].term if new_index > 1 else 0,
                entries=[new_entry],
                leader_commit=leader.commit_index,
            )
            reply = peer.handle_append_entries(args)
            if reply.success:
                ack_count += 1

        quorum = (len(self.nodes) // 2) + 1
        if ack_count >= quorum:
            leader.commit_index = new_index
            leader._apply_logs()
            # 再次通知 Follower 提交
            for peer_id in leader.peers:
                peer = self.nodes[peer_id]
                args = AppendEntriesArgs(
                    term=leader.current_term,
                    leader_id=leader.node_id,
                    prev_log_index=new_index,
                    prev_log_term=new_entry.term,
                    entries=[],
                    leader_commit=leader.commit_index,
                )
                peer.handle_append_entries(args)
            return True

        return False
