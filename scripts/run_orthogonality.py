"""信号正交性分析：知识侧(coverage) × 模型侧(self_eval/familiarity)。

目的（方案一）：不依赖 should_retrieve 标签，直接量化两路信号是否携带正交信息，
为「融合 > 单独」的互补性叙事提供上游证据。

分析三层：
  1. 相关性：Pearson + Spearman，越低越正交。
  2. 报警重叠度：各自 top-B 选出的题集 Jaccard 重叠，重叠越低越互补。
  3. 错误类型覆盖：两信号各自排名靠前的题，在「闭卷答错」上的覆盖是否不同
     （知识侧 vs 模型侧各自能兜住对方漏掉的错题）。

数据：results/mixed_batch_gt.jsonl（50 题，含 correct_closed 可作为错误标签的弱代理）
     + results/coverage_scores.json + results/model_signals_real.json
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import numpy as np
from scipy import stats

import pandas as pd


def load():
    gt = {r["id"]: r for r in (json.loads(l) for l in open("results/mixed_batch_gt.jsonl", encoding="utf-8"))}
    cov = json.load(open("results/coverage_scores.json", encoding="utf-8"))
    ms = json.load(open("results/model_signals_real.json", encoding="utf-8"))
    ids = [r["id"] for r in gt.values()]
    cov_v = np.array([cov.get(i, 0.0) for i in ids])
    se_v = np.array([ms.get(i, {}).get("self_eval", 0.5) for i in ids])
    fam_v = np.array([ms.get(i, {}).get("familiarity", 0.5) for i in ids])
    # 模型侧「该查」信号：self_eval 高→查；familiarity 低→查，所以用 1-fam
    fam_risk = 1.0 - fam_v
    closed_wrong = np.array([1 - gt[i]["correct_closed"] for i in ids])
    return ids, cov_v, se_v, fam_risk, closed_wrong


def corr_table(ids, cov, se, fam_risk):
    pairs = {
        "coverage vs self_eval": (cov, se),
        "coverage vs (1-familiarity)": (cov, fam_risk),
        "self_eval vs (1-familiarity)": (se, fam_risk),
    }
    rows = []
    for name, (a, b) in pairs.items():
        rp, pp = stats.pearsonr(a, b)
        rs, ps = stats.spearmanr(a, b)
        rows.append({"pair": name, "pearson": round(rp, 3), "pearson_p": round(pp, 3),
                     "spearman": round(rs, 3), "spearman_p": round(ps, 3)})
    return pd.DataFrame(rows)


def topk_overlap(ids, a, b, B):
    """各自 top-B 的 Jaccard 重叠（越低越互补）。"""
    top_a = set(np.argsort(-a)[:B])
    top_b = set(np.argsort(-b)[:B])
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union else 0.0, inter, len(top_a)


def main():
    ids, cov, se, fam_risk, closed_wrong = load()
    N = len(ids)
    B = max(1, int(0.2 * N))
    print(f"N={N}, B={B}")
    print(f"闭卷答错题数: {closed_wrong.sum()}/{N}")
    print()

    # 1. 相关性
    print("=== 1. 相关性（越低越正交）===")
    ct = corr_table(ids, cov, se, fam_risk)
    print(ct.to_string(index=False))
    print()

    # 2. top-B 重叠
    print("=== 2. top-B 报警重叠度（越低越互补）===")
    j1, i1, _ = topk_overlap(ids, cov, se, B)
    j2, i2, _ = topk_overlap(ids, cov, fam_risk, B)
    print(f"coverage ∩ self_eval        : Jaccard={j1:.2%} (重叠 {i1}/{B} 题)")
    print(f"coverage ∩ (1-familiarity)  : Jaccard={j2:.2%} (重叠 {i2}/{B} 题)")
    print()

    # 3. 错误类型覆盖：两信号 top-B 各自兜住多少「闭卷答错」的题，且是否互补
    print("=== 3. 错题覆盖（知识侧 vs 模型侧，各兜住多少闭卷错题）===")
    top_cov = set(np.argsort(-cov)[:B])
    top_se = set(np.argsort(-se)[:B])
    top_fam = set(np.argsort(-fam_risk)[:B])
    wrong_idx = set(np.where(closed_wrong == 1)[0])

    cov_hit = len(top_cov & wrong_idx)
    se_hit = len(top_se & wrong_idx)
    fam_hit = len(top_fam & wrong_idx)
    union_hit = len((top_cov | top_se) & wrong_idx)
    cov_only = len((top_cov - top_se) & wrong_idx)  # 知识侧独有兜住的错题
    se_only = len((top_se - top_cov) & wrong_idx)   # 模型侧独有兜住的错题

    print(f"总错题 {len(wrong_idx)} 题")
    print(f"  coverage top-B 兜住      : {cov_hit} 题")
    print(f"  self_eval top-B 兜住     : {se_hit} 题")
    print(f"  (1-fam) top-B 兜住       : {fam_hit} 题")
    print(f"  两者并集兜住             : {union_hit} 题")
    print(f"  coverage 独有兜住        : {cov_only} 题")
    print(f"  self_eval 独有兜住       : {se_only} 题")
    print()

    # 结论判定
    print("=== 判定 ===")
    if (ct["spearman"].abs() < 0.4).all():
        print("✅ 信号间 Spearman |ρ| < 0.4，相关性低 → 携带正交信息（互补性上游证据成立）")
    else:
        print("⚠️ 存在较高相关信号，正交性部分成立，需看具体 pair")
    if union_hit > max(cov_hit, se_hit):
        print(f"✅ 融合并集兜住 {union_hit} 题 > 任一单独({max(cov_hit, se_hit)})，互补有增益")
    else:
        print("⚠️ 融合并集未超单独，需进一步看")


if __name__ == "__main__":
    main()
