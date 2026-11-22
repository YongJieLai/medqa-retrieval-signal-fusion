"""信号层：coverage（稀有度 × 必要性）。

coverage(q) = Σ_{c ∈ G(q)} p_c · ν_c

- p_c : 概念 c 的缺失概率，由 IDF 校准（见 idf.py）。
- ν_c : 概念 c 的必要性，MVP 简化：
  若 c 在「题 → 答案」的唯一/最短路径上且无替代路径，ν=1；否则 ν=0。
  （完整版按 ν_c = P(E_I)·α(c) 计算，见技术交接文档第 3、4 节。）

coverage 是「答题前的估值信号」，不是答对概率（TRIAGE 已知 coverage ≠ 正确）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..schema import Question


class CoverageSignal:
    """计算每道题 coverage 分数的信号器（模型无关，只读 KG/语料）。"""

    def __init__(self, idf_map: Dict[str, float], tau: float = 1.0):
        # idf_map: 概念 -> idf 值。idf_median 取所有概念 idf 的中位数（代码纪律 §4）。
        self.idf_map = idf_map
        self.tau = tau
        self.idf_median = self._compute_median(idf_map.values())

    @staticmethod
    def _compute_median(values) -> float:
        vs = sorted(values)
        n = len(vs)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return vs[n // 2]
        return (vs[n // 2 - 1] + vs[n // 2]) / 2.0

    def missing_prob(self, concept: str) -> float:
        """p_c = σ((idf(c) - idf_median) / τ)。未收录概念按 idf=0 处理。"""
        idf = self.idf_map.get(concept, 0.0)
        from .idf import missing_prob as _mp

        return _mp(idf, self.idf_median, self.tau)

    def necessity(self, concept: str, question: Question, kg_shortest_paths: Dict[str, set]) -> float:
        """MVP 必要性 ν_c。

        kg_shortest_paths: 概念 -> 该概念是否位于「题→答案」唯一/最短路径且无替代。
        这里由外部预计算好（数据构造阶段），MVP 下返回 0/1。
        完整版会换成 ν_c = P(E_I)·α(c)。
        """
        return 1.0 if kg_shortest_paths.get(concept, set()) else 0.0

    def coverage(self, question: Question, kg_shortest_paths: Dict[str, set]) -> float:
        """coverage(q) = Σ_c p_c · ν_c。"""
        total = 0.0
        for c in question.concepts:
            total += self.missing_prob(c) * self.necessity(c, question, kg_shortest_paths)
        return total
