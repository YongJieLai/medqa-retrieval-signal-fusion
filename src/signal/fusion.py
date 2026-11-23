"""信号层：融合信号（知识侧 × 模型侧互补）。

主叙事（H1 修正后的稳健立足点）：
    知识侧信号（coverage：稀有度 × 必要性）捕捉「缺知识 / 罕见病」型错误；
    模型侧信号（self_eval 自评 / familiarity 熟悉度）捕捉「推理不确定 / 判断失败」型错误。
    两者覆盖不同的错误类型 → 融合后漏检率低于任一单独信号。

融合方式（MVP，无需训练，可解释）：
    1. rank_max    : 两信号各自归一化到 [0,1]，取逐题最大值（"任一信号报警即查"，保召回）。
    2. rank_sum    : 两信号归一化后相加（"证据叠加"）。
    3. min_rank    : 取两信号各自排序的最小秩（更稳健的「任一信号报警」）。

纪律说明：本模块是「答题前估值信号」，不读 answer；归一化基于当前批次的
    分数分布（跨题），不引入 ground truth，因此不构成泄漏。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..schema import Question


def _minmax_norm(scores: Sequence[float]) -> List[float]:
    """min-max 归一化到 [0,1]。全同值退化为 0.5（无信息）。"""
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _rank_norm(scores: Sequence[float]) -> List[float]:
    """按降序排名归一化到 [0,1]（分位数 rank，稳健，不受量纲影响）。"""
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    rank = [0.0] * n
    for r, i in enumerate(order):
        rank[i] = 1.0 - r / max(1, n - 1)  # 最高分 → 1.0，最低 → 0.0
    return rank


def _scores_by_qid(questions: Sequence[Question], scores: Sequence[float]) -> Dict[str, float]:
    return {q.question_id: s for q, s in zip(questions, scores)}


def fuse(
    questions: Sequence[Question],
    knowledge_scores: Sequence[float],
    model_scores: Sequence[float],
    mode: str = "rank_max",
) -> List[float]:
    """融合知识侧与模型侧信号，返回逐题融合分数（越高越该查）。

    mode:
      - rank_max : 归一化后取 max（任一信号报警即查，保召回，主推）。
      - rank_sum : 归一化后相加（证据叠加）。
      - min_rank : 取两信号排名的更优（min）秩，稳健的「任一报警」。
    """
    kn = _rank_norm(knowledge_scores)
    mn = _rank_norm(model_scores)

    if mode == "rank_sum":
        return [a + b for a, b in zip(kn, mn)]
    if mode == "min_rank":
        order_k = sorted(range(len(kn)), key=lambda i: -kn[i])
        order_m = sorted(range(len(mn)), key=lambda i: -mn[i])
        rank_k = {i: r for r, i in enumerate(order_k)}
        rank_m = {i: r for r, i in enumerate(order_m)}
        n = len(kn)
        # 取更靠前的秩（越小越好），再转回分数（越大越好）
        return [1.0 - min(rank_k[i], rank_m[i]) / max(1, n - 1) for i in range(n)]
    # rank_max（默认）
    return [max(a, b) for a, b in zip(kn, mn)]
