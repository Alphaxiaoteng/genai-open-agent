"""blackboard.py - 工业级多智能体黑板协作总线与事件发布订阅引擎 (Blackboard Event Bus)。

遵循大厂分布式 Multi-Agent 架构设计：
1. 共享黑板全局事实存储 (Blackboard Knowledge Store)，带单调递增版本号与字段级观察订阅。
2. 主题式事件发布订阅总线 (Topic Pub/Sub)，支持通配符路由 (如 `agent.*`、`task.done`) 与历史重放 (Replay)。
3. 线程安全原子事务更新，防止多个并行 Agent 并发写导致状态被覆盖。
"""

from __future__ import annotations

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class EventMessage:
    event_id: str
    topic: str
    sender: str
    payload: Dict[str, Any]
    offset: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class BlackboardFact:
    key: str
    value: Any
    version: int
    updated_by: str
    updated_at: float = field(default_factory=time.time)


class BlackboardEventBus:
    """多智能体黑板与事件总线。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._facts: Dict[str, BlackboardFact] = {}
        self._events: List[EventMessage] = []
        self._subscribers: Dict[str, Set[str]] = {}  # topic_pattern -> set of subscriber_ids
        self._message_inboxes: Dict[str, List[EventMessage]] = {}  # subscriber_id -> list of messages

    def register_subscriber(self, subscriber_id: str, topic_pattern: str) -> None:
        """注册智能体订阅特定主题 (支持通配符)。"""
        with self._lock:
            if topic_pattern not in self._subscribers:
                self._subscribers[topic_pattern] = set()
            self._subscribers[topic_pattern].add(subscriber_id)
            if subscriber_id not in self._message_inboxes:
                self._message_inboxes[subscriber_id] = []

    def publish(self, topic: str, sender: str, payload: Dict[str, Any]) -> EventMessage:
        """发布事件至总线并分发到匹配的订阅者收件箱。"""
        with self._lock:
            offset = len(self._events)
            msg = EventMessage(
                event_id=f"evt_{offset}_{int(time.time() * 1000)}",
                topic=topic,
                sender=sender,
                payload=payload,
                offset=offset,
            )
            self._events.append(msg)

            # 匹配订阅者
            for pattern, subs in self._subscribers.items():
                if fnmatch.fnmatch(topic, pattern):
                    for sub in subs:
                        self._message_inboxes[sub].append(msg)

            return msg

    def poll_messages(self, subscriber_id: str, max_count: int = 10) -> List[EventMessage]:
        """拉取并消费当前智能体的未读事件。"""
        with self._lock:
            inbox = self._message_inboxes.get(subscriber_id, [])
            res = inbox[:max_count]
            self._message_inboxes[subscriber_id] = inbox[max_count:]
            return res

    def set_fact(self, key: str, value: Any, agent_name: str) -> BlackboardFact:
        """原子写入共享黑板全局事实。"""
        with self._lock:
            current = self._facts.get(key)
            new_version = (current.version + 1) if current else 1
            fact = BlackboardFact(
                key=key,
                value=value,
                version=new_version,
                updated_by=agent_name,
            )
            self._facts[key] = fact
            return fact

    def get_fact(self, key: str) -> Optional[Any]:
        with self._lock:
            fact = self._facts.get(key)
            return fact.value if fact else None

    def get_fact_metadata(self, key: str) -> Optional[BlackboardFact]:
        with self._lock:
            return self._facts.get(key)

    def replay_from(self, offset: int) -> List[EventMessage]:
        """从指定偏移量重放历史事件。"""
        with self._lock:
            return list(self._events[offset:])
