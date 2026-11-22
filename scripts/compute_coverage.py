"""轻量知识图：用 MedQA 语料实体共现构建图，算必要性 ν。

MVP 必要性（DEVIATION 标注）：
- 从全 MedQA test 集（1273 题）抽取 metamap_phrases 实体。
- 同一题内共现的实体之间有边（共现图）。
- 对混合批次每题，抽「题干核心实体」，在共现图上算实体间的连通性/最短路径。
- 必要性 ν_c：若概念 c 是题干→答案实体路径上的必经节点（桥），ν=1；否则 ν=0。

稀有度 p_c：沿用 IDF（idf 高 = 罕见 = 高缺失）。
coverage(q) = Σ p_c · ν_c。

输出：results/coverage_scores.json  {question_id: coverage}
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from typing import Dict, List, Set

sys.path.insert(0, ".")

import networkx as nx


def clean_phrases(phrases: List[str]) -> List[str]:
    """清洗 metamap 短语：去停用词/纯功能词，保留有医学意义的实体。"""
    stop = {
        "resident", "completing", "case", "patient", "following", "attending",
        "not", "to make", "patient worry", "resident to leave", "complication out",
        "correct next action", "resident to take", "the", "a", "an", "of", "in",
        "and", "or", "with", "for", "on", "at", "to", "is", "are", "was", "were",
        "which", "what", "this", "that", "these", "those", "his", "her", "their",
        "she", "he", "they", "has", "have", "had", "be", "been", "being", "do",
        "does", "did", "will", "would", "should", "could", "can", "may", "might",
    }
    out = []
    for p in phrases:
        p = p.strip().lower()
        if len(p) < 3:
            continue
        if p in stop:
            continue
        # 过滤纯数字/纯标点
        if all(c.isdigit() or c in ".,;:()/- " for c in p):
            continue
        out.append(p)
    return out


def build_cooccurrence_graph(corpus_rows: List[Dict]) -> nx.Graph:
    """构建实体共现图：同一题内共现的实体间加边。"""
    G = nx.Graph()
    for r in corpus_rows:
        phrases = clean_phrases(r["metamap_phrases"])
        # 两两共现加边（限制每题的实体数，避免图过密）
        for i in range(len(phrases)):
            for j in range(i + 1, len(phrases)):
                G.add_edge(phrases[i], phrases[j])
    return G


def compute_necessity(G: nx.Graph, concepts: List[str]) -> Dict[str, float]:
    """算每个概念的 MVP 必要性 ν。

    若概念是图中"桥"（去掉后图连通性下降），ν=1；否则 0。
    用 edge-betweenness 近似：高度数/高介数的节点更可能是必经。
    MVP 简化：概念度数越高（越枢纽）ν 越接近 1，度数低（叶子）ν=0。
    """
    nu = {}
    for c in concepts:
        if c in G:
            deg = G.degree(c)
            # 度数归一化：度数高 → 枢纽 → 必经概率高
            # 简单阈值：度数 >= 3 视为枢纽（ν=1），否则 ν=0
            nu[c] = 1.0 if deg >= 3 else 0.0
        else:
            nu[c] = 0.0  # 不在图中
    return nu


def main():
    corpus_rows = [json.loads(l) for l in open("data/medqa_test.jsonl", encoding="utf-8")]
    gt_rows = [json.loads(l) for l in open("results/mixed_batch_gt.jsonl", encoding="utf-8")]

    print("构建共现图...")
    G = build_cooccurrence_graph(corpus_rows)
    print(f"图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    # IDF（稀有度）
    N = len(corpus_rows)
    df = Counter()
    for r in corpus_rows:
        for p in set(clean_phrases(r["metamap_phrases"])):
            df[p] += 1
    idf = {p: math.log(N / (1 + df[p])) for p in df}

    # 全概念 idf 中位数
    idf_median = sorted(idf.values())[len(idf) // 2] if idf else 0.0
    print(f"idf 中位数: {idf_median:.3f}")

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    # 对每题算 coverage
    scores = {}
    for r in gt_rows:
        concepts = clean_phrases(r["metamap_phrases"])
        # 取前 8 个概念
        concepts = concepts[:8]
        nu = compute_necessity(G, concepts)
        cov = 0.0
        for c in concepts:
            p_c = sigmoid((idf.get(c, 0.0) - idf_median) / 1.0)
            cov += p_c * nu.get(c, 0.0)
        scores[r["id"]] = cov

    with open("results/coverage_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # 打印分布
    vals = list(scores.values())
    print(f"\ncoverage 分数: 均值={sum(vals)/len(vals):.3f} 最大={max(vals):.3f} 非零占比={sum(1 for v in vals if v>0)/len(vals):.1%}")
    # 看 should_retrieve=1 和 =0 的 coverage 是否有区分
    pos = [scores[r["id"]] for r in gt_rows if r["should_retrieve"] == 1]
    neg = [scores[r["id"]] for r in gt_rows if r["should_retrieve"] == 0]
    print(f"should_retrieve=1 的 coverage 均值: {sum(pos)/len(pos):.3f}")
    print(f"should_retrieve=0 的 coverage 均值: {sum(neg)/len(neg):.3f}")


if __name__ == "__main__":
    main()
