"""统一数据 schema。

代码纪律第 7 节：题目输入统一 schema；answer 仅执行层/评测层可读，
信号层与分配层的数据对象中不得带 answer（答题前约束，防泄漏）。

因此这里拆成两个对象：
- Question : 信号层/分配层可见，不含 answer。
- LabeledQuestion : 执行层/评测层可见，含 answer 与 ground truth。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

# 题型标签
QType = Literal["type1", "type2"]  # type1=罕见病(高缺失×低推理), type2=常见病推理(低缺失×高推理)

# 检索决策
Action = Literal["closed", "retrieve"]


@dataclass
class Question:
    """信号层/分配层可见的题目（不含 answer）。"""

    question_id: str
    stem: str
    options: List[str]  # 长度为 4
    qtype: QType
    # 题干锚定的概念实体（由构造阶段给出，MVP 用精确匹配），
    # 概念召回层用它算稀有度/必要性。
    concepts: List[str] = field(default_factory=list)


@dataclass
class LabeledQuestion(Question):
    """执行层/评测层可见的题目（含答案与 ground truth）。"""

    answer: str = ""              # 正确选项，如 "A"
    correct_closed: int = 0       # 闭卷作答是否正确
    correct_open: int = 0         # 开卷作答是否正确
    delta: int = 0                # correct_open - correct_closed，负值按 0 处理
    should_retrieve: int = 0      # 1 当且仅当 delta == +1


@dataclass
class Decision:
    """分配层输出：每道题的检索决策。"""

    question_id: str
    action: Action
    score: float = 0.0  # 连续分数（coverage / baseline 分数），用于排序
