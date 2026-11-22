"""分配层：跨题全局预算分配。

价值函数 F(S) = Σ_{q∈S} coverage(q) 单调子模。
在总预算 B 下最大化 F(S)，子模贪心可证 1-1/e。
MVP：各题检索成本统一为 1，贪心退化为「按 coverage 降序取前 B 道」。

决策输出 schema：{question_id, action ∈ {closed, retrieve}}（代码纪律 §7）。
"""
from __future__ import annotations

from typing import List, Sequence

from ..schema import Decision, Question


def allocate_top_b(
    questions: Sequence[Question],
    scores: Sequence[float],
    budget: int,
    threshold: bool = False,
    threshold_value: float = 0.5,
) -> List[Decision]:
    """MVP 分配。

    - threshold=False：按分数降序取前 B 道标为 retrieve（连续排序方法）。
    - threshold=True：按阈值判「查/不查」（二元方法，Marina/QuCo/MedCoG 用），
      可能超支或不足，但预算约束下会裁剪到 B（代码纪律 §9 红线：预算全程生效）。
    """
    n = len(questions)
    decisions: List[Decision] = []

    if threshold:
        # 二元阈值方法：分数 >= threshold_value → retrieve
        flagged = [i for i in range(n) if scores[i] >= threshold_value]
        # 预算裁剪：若超支，按分数降序取前 B；若不足，保持不足（记录为二元方法不自洽）
        if len(flagged) > budget:
            flagged = sorted(flagged, key=lambda i: -scores[i])[:budget]
        retrieve_ids = set(flagged)
    else:
        order = sorted(range(n), key=lambda i: -scores[i])
        retrieve_ids = set(order[:budget])

    for i, q in enumerate(questions):
        action = "retrieve" if i in retrieve_ids else "closed"
        decisions.append(Decision(question_id=q.question_id, action=action, score=scores[i]))
    return decisions


def submodular_greedy(
    questions: Sequence[Question],
    scores: Sequence[float],
    budget: int,
    costs: Sequence[float] | None = None,
) -> List[Decision]:
    """完整子模贪心（预算 B 下选边际增益最大的题）。

    当各题检索成本统一（costs=None）时，与 top-B 排序等价。
    提供 costs 时按「边际增益 / 成本」贪心（近似处理，成本统一时精确）。
    """
    n = len(questions)
    if costs is None:
        costs = [1.0] * n

    selected: List[int] = []
    remaining = set(range(n))
    remaining_budget = float(budget)

    # 子模贪心：每次选边际增益/成本最大的题
    while remaining and remaining_budget > 0:
        best_i = -1
        best_ratio = -1.0
        for i in remaining:
            if costs[i] <= remaining_budget + 1e-9:
                ratio = scores[i] / costs[i] if costs[i] > 0 else scores[i]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        remaining.remove(best_i)
        remaining_budget -= costs[best_i]

    retrieve_ids = set(selected)
    decisions = []
    for i, q in enumerate(questions):
        action = "retrieve" if i in retrieve_ids else "closed"
        decisions.append(Decision(question_id=q.question_id, action=action, score=scores[i]))
    return decisions
