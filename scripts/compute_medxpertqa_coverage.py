"""第二步：计算 MedXpertQA 200 题的知识侧 coverage 信号。

coverage(q) = Σ_c p_c · ν_c
- p_c（稀有度）：实体 c 在 PrimeKG 的度数（degree），度数低 = 罕见 = 知识缺口大。
  用 IDF 风格：p_c = log(N_total / (1 + degree_c))，N_total 为 PrimeKG 实体总数。
- ν_c（必要性）：MVP 简化为 1（LLM 抽出的都是题干核心医学概念，都「必要」）。

同时记录实体在 PrimeKG 的「命中率」（多少实体能找到），作为 coverage 的补充质量指标。

输出：results/medxpertqa_coverage.json  {id: coverage}
"""
from __future__ import annotations

import json
import math

ENTITIES = "results/medxpertqa_entities.jsonl"
DEGREE = "kg/primekg/entity_degree.json"
OUT = "results/medxpertqa_coverage.json"


def main():
    degree = json.load(open(DEGREE, encoding="utf-8"))
    N_total = len(degree)  # PrimeKG 唯一实体总数
    print(f"PrimeKG 唯一实体数: {N_total}")

    rows = [json.loads(l) for l in open(ENTITIES, encoding="utf-8")]
    scores = {}
    detail = {}

    for r in rows:
        entities = [e.strip().lower() for e in r["entities"]]
        cov = 0.0
        hit = 0
        for e in entities:
            d = degree.get(e, 0)
            if d > 0:
                # 稀有度：度数低 → IDF 高 → 知识缺口大
                p_c = math.log(N_total / (1 + d))
                cov += p_c
                hit += 1
            # 实体不在 PrimeKG 里：视为「极罕见」（知识缺口最大），但 MVP 先不计入，
            # 避免 LLM 抽取噪声（如 'blood pressure' 这类非规范化术语）干扰。
        # 平均稀有度（而非求和），避免实体数量差异影响可比性
        if hit > 0:
            cov = cov / hit
        scores[r["id"]] = cov
        detail[r["id"]] = {"coverage": cov, "hit": hit, "total": len(entities)}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # 分布统计
    vals = list(scores.values())
    hits = [d["hit"] for d in detail.values()]
    totals = [d["total"] for d in detail.values()]
    print(f"\ncoverage 分布: 均值={sum(vals)/len(vals):.3f} 最小={min(vals):.3f} 最大={max(vals):.3f}")
    print(f"PrimeKG 命中率: 平均 {sum(hits)/len(hits):.2f} / {sum(totals)/len(totals):.2f} 个实体")
    print(f"完全未命中（coverage=0）的题: {sum(1 for v in vals if v==0)}/{len(vals)}")

    # 看 coverage 与闭卷答错、检索增益的关系
    p1 = {json.loads(l)["id"]: json.loads(l) for l in open("results/p1_m0_m1.jsonl", encoding="utf-8")}
    p2 = {json.loads(l)["id"]: json.loads(l) for l in open("results/p2_semantic.jsonl", encoding="utf-8")}
    print("\n=== coverage 与「闭卷答错」「检索增益」的关系 ===")
    import numpy as np
    covs, m0_wrong, gains = [], [], []
    for rid, cov in scores.items():
        if cov == 0:
            continue
        m0_ok = p1[rid]["M0_pred"] == p1[rid]["label"]
        m5_ok = p2[rid]["M5_pred"] == p2[rid]["label"]
        gain = 1 if (not m0_ok and m5_ok) else (-1 if (m0_ok and not m5_ok) else 0)
        covs.append(cov); m0_wrong.append(0 if m0_ok else 1); gains.append(gain)
    print(f"  corr(coverage, 闭卷答错) = {np.corrcoef(covs, m0_wrong)[0,1]:+.3f}")
    print(f"  corr(coverage, 检索增益) = {np.corrcoef(covs, gains)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
