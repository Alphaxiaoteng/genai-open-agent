"""critic.py - 智能体自主反思与批判改进闭环 (Self-Reflection Critic Loop)。

构建生成者 (Generator) 与批评者 (Critic) 的对抗迭代：
1. 采用多维度量规评分卡 (Rubric Score Card):
   - 准确性 (Accuracy)
   - 完备性 (Completeness)
   - 简洁性 (Conciseness)
   - 安全合规性 (Safety)
2. 设定达标及格线 (pass_threshold 如 85分)，未达标时提取具体改进指令，迭代反馈直至达标或达到最大迭代上限。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class CritiqueReport:
    iteration: int
    accuracy_score: float
    completeness_score: float
    conciseness_score: float
    safety_score: float
    critique_notes: List[str]
    suggested_revision: str

    @property
    def composite_score(self) -> float:
        return round(
            self.accuracy_score * 0.4
            + self.completeness_score * 0.3
            + self.conciseness_score * 0.15
            + self.safety_score * 0.15,
            2,
        )


@dataclass
class ReflectionResult:
    final_output: str
    iterations_run: int
    is_passed: bool
    final_score: float
    history: List[CritiqueReport]


class SelfReflectionCriticLoop:
    """自主反思与批判改进循环。"""

    def __init__(self, pass_threshold: float = 85.0, max_rounds: int = 3) -> None:
        self.pass_threshold = pass_threshold
        self.max_rounds = max_rounds

    def execute_loop(
        self,
        initial_prompt: str,
        generate_fn: Callable[[str, Optional[str]], str],
        critique_fn: Callable[[str, str], CritiqueReport],
    ) -> ReflectionResult:
        """执行生成-反思循环。"""
        current_draft = generate_fn(initial_prompt, None)
        history: List[CritiqueReport] = []

        for round_idx in range(1, self.max_rounds + 1):
            report = critique_fn(initial_prompt, current_draft)
            report.iteration = round_idx
            history.append(report)

            if report.composite_score >= self.pass_threshold:
                return ReflectionResult(
                    final_output=current_draft,
                    iterations_run=round_idx,
                    is_passed=True,
                    final_score=report.composite_score,
                    history=history,
                )

            # 未达标，使用改进建议重新生成
            feedback = "; ".join(report.critique_notes)
            current_draft = generate_fn(initial_prompt, feedback)

        # 达到最大轮次仍未达标
        last_score = history[-1].composite_score if history else 0.0
        return ReflectionResult(
            final_output=current_draft,
            iterations_run=self.max_rounds,
            is_passed=last_score >= self.pass_threshold,
            final_score=last_score,
            history=history,
        )
