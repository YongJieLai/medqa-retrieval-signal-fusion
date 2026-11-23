"""第三步：coverage × familiarity 正交性 + 融合门控实验。

命题：两路信号（知识侧 coverage、模型侧 familiarity）单独都无法预测「该不该检索」，
但正交互补，融合后（并集/联合门控）优于单独。

分析：
1. 相关性（正交性）
2. top-B 报警重叠（Jaccard，越低越互补）
3. 错题覆盖：各自 top-B 兜住多少「闭卷答错」的题，并集是否 > 单独
4. 融合门控：单信号门控 vs 双信号联合门控的准确率对比

数据（全部离线）：
- coverage: results/medxpertqa_coverage.json
- familiarity: results/medcog_signals.jsonl
- M0 闭卷: results/p1_m0_m1.jsonl
- M5 语义检索: results/p2_semantic.jsonl
"""
from __future__ import annotations

import json
import numpy as np


def load():
    cov = json.load(open("results/medxpertqa_coverage.json", encoding="utf-8"))
    sig = {json.loads(l)["id"]: json.loads(l) for l in open("results/medcog_signals.jsonl", encoding="utf-8")}
    p1 = {json.loads(l)["id"]: json.loads(l) for l in open("results/p1_m0_m1.jsonl", encoding="utf-8")}
    p2 = {json.loads(l)["id"]: json.loads(l) for l in open("results/p2_semantic.jsonl", encoding="utf-8")}
    return cov, sig, p1, p2


def main():
    cov, sig, p1, p2 = load()
    ids = [i for i in cov if cov[i] > 0]  # 排除 coverage=0
    N = len(ids)

    cov_v = np.array([cov[i] for i in ids])
    fam_v = np.array([sig[i]["familiarity"] for i in ids])
    fam_risk = 1.0 - fam_v  # 不熟悉 = 该检索（familiarity 低）

    labels = {i: p1[i]["label"] for i in ids}
    m0 = {i: p1[i]["M0_pred"] for i in ids}
    m5 = {i: p2[i]["M5_pred"] for i in ids}

    # 基线
    acc_m0 = sum(1 for i in ids if m0[i] == labels[i]) / N
    acc_m5 = sum(1 for i in ids if m5[i] == labels[i]) / N
    print(f"N={N}（coverage>0）")
    print(f"基线：M0 全闭卷={acc_m0:.4f}  M5 全检索={acc_m5:.4f}\n")

    # 1. 正交性
    print("=== 1. 正交性 ===")
    print(f"  corr(coverage, 1-familiarity) = {np.corrcoef(cov_v, fam_risk)[0,1]:+.3f}")

    # 2. top-B 重叠
    B = max(1, int(0.2 * N))
    top_cov = set(np.argsort(-cov_v)[:B])
    top_fam = set(np.argsort(-fam_risk)[:B])  # familiarity 低 = 该检索
    jaccard = len(top_cov & top_fam) / len(top_cov | top_fam)
    print(f"\n=== 2. top-{B} 报警重叠 ===")
    print(f"  coverage ∩ (1-familiarity): Jaccard={jaccard:.2%}（重叠 {len(top_cov & top_fam)}/{B}）")

    # 3. 错题覆盖
    wrong_idx = set(i for i in range(N) if m0[ids[i]] != labels[ids[i]])
    cov_hit = len(top_cov & wrong_idx)
    fam_hit = len(top_fam & wrong_idx)
    union_hit = len((top_cov | top_fam) & wrong_idx)
    cov_only = len((top_cov - top_fam) & wrong_idx)
    fam_only = len((top_fam - top_cov) & wrong_idx)
    print(f"\n=== 3. 错题覆盖（闭卷答错 {len(wrong_idx)} 题）===")
    print(f"  coverage 单独兜住: {cov_hit}")
    print(f"  1-familiarity 单独兜住: {fam_hit}")
    print(f"  并集兜住: {union_hit}  (coverage独有 {cov_only} + 模型侧独有 {fam_only})")
    print(f"  → 互补性: 并集{union_hit} vs 单独最大{max(cov_hit, fam_hit)}，增益 {union_hit - max(cov_hit, fam_hit)}")

    # 4. 融合门控
    print(f"\n=== 4. 门控对比（单信号 vs 双信号融合）===")

    def gating_acc(tau_cov, tau_fam, mode="or"):
        """mode=or: cov>tau_cov 或 fam<tau_fam 就检索；and: 两者都满足才检索"""
        correct = 0
        for idx, i in enumerate(ids):
            need_cov = cov_v[idx] > tau_cov  # 知识缺口大
            need_fam = fam_v[idx] < tau_fam  # 不熟悉
            if mode == "or":
                retrieve = need_cov or need_fam
            else:
                retrieve = need_cov and need_fam
            pred = m5[i] if retrieve else m0[i]
            if pred == labels[i]:
                correct += 1
        return correct / N

    # 单信号：coverage 门控（coverage 高才检索）
    best_cov = (0, None)
    for tau in np.percentile(cov_v, np.arange(20, 90, 10)):
        a = gating_acc(tau, 2.0, "or")  # tau_fam=2 即 fam<2 恒真，只由 coverage 决定
        if a > best_cov[0]:
            best_cov = (a, tau)

    # 单信号：familiarity 门控（fam 低才检索）
    best_fam = (0, None)
    for tau in np.percentile(fam_v, np.arange(10, 90, 10)):
        a = gating_acc(1e9, tau, "or")  # tau_cov=1e9 即 cov>1e9 恒假，只由 fam 决定
        if a > best_fam[0]:
            best_fam = (a, tau)

    # 双信号融合（OR：任一信号报警就检索）
    best_or = (0, None, None)
    for tc in np.percentile(cov_v, np.arange(20, 90, 10)):
        for tf in np.percentile(fam_v, np.arange(10, 90, 10)):
            a = gating_acc(tc, tf, "or")
            if a > best_or[0]:
                best_or = (a, tc, tf)

    print(f"  单信号 coverage 门控: 最优 {best_cov[0]:.4f}（τ_cov={best_cov[1]:.2f}）")
    print(f"  单信号 familiarity 门控: 最优 {best_fam[0]:.4f}（τ_fam={best_fam[1]:.2f}）")
    print(f"  双信号 OR 融合门控: 最优 {best_or[0]:.4f}（τ_cov={best_or[1]:.2f}, τ_fam={best_or[2]:.2f}）")
    print(f"\n  对比：融合 {best_or[0]:.4f} vs 单信号最佳 {max(best_cov[0], best_fam[0]):.4f}，"
          f"相对提升 {best_or[0] - max(best_cov[0], best_fam[0]):+.4f}")

    # 保存正交性摘要
    summary = {
        "N": N,
        "B": B,
        "n_wrong": len(wrong_idx),
        "corr_cov_fam_risk": round(np.corrcoef(cov_v, fam_risk)[0, 1], 3),
        "jaccard_topk": round(jaccard, 3),
        "cov_hit": cov_hit,
        "fam_hit": fam_hit,
        "union_hit": union_hit,
        "cov_only": cov_only,
        "fam_only": fam_only,
        "best_gating_cov": round(best_cov[0], 4),
        "best_gating_fam": round(best_fam[0], 4),
        "best_gating_or": round(best_or[0], 4),
    }
    with open("results/orthogonality_medxpertqa.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 results/orthogonality_medxpertqa.json")


if __name__ == "__main__":
    main()
