"""信号层：各 baseline（模型侧 + 知识侧）。

- M1 · MedCoG 自评：模型对每题输出「是否需要检索」的自评分数（需要调用 LLM）。
- M2 · KBM 熟悉度：用模型对该题的熟悉度代理（平均 token logprob），低熟悉度 → 判查。
- M3 · Marina 实体流行度：抽题干核心实体，按实体 IDF/流行度打分，低频 → 判查。
- M4 · QuCo-RAG 频率 + 共现：实体频率低 / 零共现 → 判缺知识 → 判查。

M3/M4 是「知识侧」，不读模型内部，可直接用 idf/共现数据算。
M1/M2 是「模型侧」，需要读模型内部状态（自评分数 / logprob）。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..schema import Question


class MarinaSignal:
    """M3 · Marina（实体流行度，规则版）。

    抽题干核心实体，按实体 IDF 打分，IDF 高（低频/罕见）→ 判「查」。
    输出连续分数 = 题干实体 IDF 的最大值（代表「最罕见实体」的缺知识程度）。
    """

    def __init__(self, idf_map: Dict[str, float]):
        self.idf_map = idf_map

    def score(self, question: Question) -> float:
        if not question.concepts:
            return 0.0
        return max(self.idf_map.get(c, 0.0) for c in question.concepts)


class QuCoRAGSignal:
    """M4 · QuCo-RAG（频率 + 共现，规则版）。

    按「实体频率 + 三元组共现」判断是否缺知识：
    低频实体 / 零共现 → 判缺知识 → 判查。
    输出连续分数 = -min(共现计数)，或题干实体 IDF 平均（无共现时）。
    MVP：用实体 IDF 平均 + 共现缺失惩罚近似。
    """

    def __init__(self, idf_map: Dict[str, float], cooccur: Dict[str, int]):
        self.idf_map = idf_map
        self.cooccur = cooccur

    def score(self, question: Question) -> float:
        if not question.concepts:
            return 0.0
        # 共现缺失惩罚：无共现的概念越多，越缺知识
        missing = sum(1 for c in question.concepts if self.cooccur.get(c, 0) == 0)
        avg_idf = sum(self.idf_map.get(c, 0.0) for c in question.concepts) / len(question.concepts)
        return avg_idf + missing  # 简单线性组合，MVP 够用


class MedCoGSelfEvalSignal:
    """M1 · MedCoG 自评（模型侧）。

    需要 LLM 对每题输出「是否需要检索」的自评分数。
    这里通过注入 self_eval_scores（预先由 LLM 计算好）提供。
    自评分数高 → 判「查」（模型认为自己不确定）。
    """

    def __init__(self, self_eval_scores: Dict[str, float]):
        self.self_eval_scores = self_eval_scores

    def score(self, question: Question) -> float:
        return self.self_eval_scores.get(question.question_id, 0.0)


class KBMFamiliaritySignal:
    """M2 · KBM 熟悉度（模型侧）。

    用模型对该题的熟悉度代理（平均 token logprob / perplexity）。
    低熟悉度 → 判「查」。输出分数 = 1 - familiarity（越高越该查）。
    这里通过注入 familiarity（预先由 LLM 计算好）提供。
    """

    def __init__(self, familiarity: Dict[str, float]):
        self.familiarity = familiarity

    def score(self, question: Question) -> float:
        f = self.familiarity.get(question.question_id, 1.0)
        return 1.0 - f
