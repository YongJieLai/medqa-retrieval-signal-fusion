"""稀有度 → 缺失概率。

代码纪律第 4 节公式：
- idf(c) = log(N / (1 + df(c)))，N 为语料总文档数，df(c) 为含概念 c 的文档数。
- 缺失概率 p_c = σ((idf(c) - idf_median) / τ)，σ 为 sigmoid，τ 初始 1.0。

注意 σ(x)=1/(1+e^{-x}) 的数值稳定实现（x<0 时用等价变换避免上溢）。
"""
from __future__ import annotations

import math
from typing import Dict, Sequence


def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def idf_from_df(df: int, N: int) -> float:
    """idf(c) = log(N / (1 + df(c)))。df=0 时取 N（即 idf=log(N/(1+N))，接近 0）。"""
    if N <= 0:
        raise ValueError("N 必须为正")
    return math.log(N / (1 + df))


def median(values: Sequence[float]) -> float:
    """中位数。"""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def missing_prob(idf: float, idf_median: float, tau: float = 1.0) -> float:
    """p_c = σ((idf(c) - idf_median) / τ)。"""
    return sigmoid((idf - idf_median) / tau)


def compute_idf(df_map: Dict[str, int], N: int) -> Dict[str, float]:
    """对每个概念 c 计算 idf(c)。"""
    return {c: idf_from_df(d, N) for c, d in df_map.items()}
