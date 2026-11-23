"""调控信号门控实验：用 familiarity（模型侧信号）判断「该不该检索」。

核心问题：能否用调控信号避免"检索帮倒忙"，放大 M5 过滤的正增益？

门控策略：
- G-fam：familiarity < τ → 检索 + M5 过滤作答；否则闭卷直接答。
- （后续可扩展 G-cov 知识侧、G-fusion 双信号）

数据全部离线复用（无新 LLM 调用）：
- familiarity：results/medcog_signals.jsonl
- M0 闭卷预测：results/p1_m0_m1.jsonl
- M5 语义检索预测：results/p2_semantic.jsonl

输出：门控结果 + 最优阈值 + 与基线对比。
"""
from __future__ import annotations

import json
import numpy as np


def load_dict(path, key="id"):
    return {json.loads(l)[key]: json.loads(l) for l in open(path, encoding="utf-8")}


def main():
    sig = load_dict("results/medcog_signals.jsonl")
    p1 = load_dict("results/p1_m0_m1.jsonl")
    p2 = load_dict("results/p2_semantic.jsonl")

    ids = list(sig.keys())
    labels = {i: p1[i]["label"] for i in ids}
    m0 = {i: p1[i]["M0_pred"] for i in ids}
    m5 = {i: p2[i]["M5_pred"] for i in ids}
    fam = {i: sig[i]["familiarity"] for i in ids}

    # 基线
    n = len(ids)
    acc_m0 = sum(1 for i in ids if m0[i] == labels[i]) / n
    acc_m5 = sum(1 for i in ids if m5[i] == labels[i]) / n
    print(f"基线：M0 全闭卷 = {acc_m0:.4f}，M5 全检索 = {acc_m5:.4f}\n")

    # 门控：familiarity < τ → 检索(M5)，否则闭卷(M0)
    print("=== G-fam 门控（familiarity < τ 才检索）===")
    print(f"{'阈值τ':>8} {'检索题数':>8} {'闭卷题数':>8} {'门控准确率':>10} {'vs M0':>8} {'vs M5':>8}")
    best = None
    for tau in np.arange(0.30, 0.96, 0.05):
        tau = round(tau, 2)
        retrieve_n = 0
        closed_n = 0
        correct = 0
        for i in ids:
            if fam[i] < tau:
                retrieve_n += 1
                pred = m5[i]
            else:
                closed_n += 1
                pred = m0[i]
            if pred == labels[i]:
                correct += 1
        acc_g = correct / n
        print(f"{tau:>8.2f} {retrieve_n:>8d} {closed_n:>8d} {acc_g:>10.4f} {acc_g-acc_m0:>+8.4f} {acc_g-acc_m5:>+8.4f}")
        if best is None or acc_g > best[0]:
            best = (acc_g, tau, retrieve_n, closed_n)

    print(f"\n最优阈值 τ={best[1]}：准确率 {best[0]:.4f}（检索 {best[2]} 题，闭卷 {best[3]} 题）")
    print(f"  相对 M0 全闭卷：{best[0]-acc_m0:+.4f}")
    print(f"  相对 M5 全检索：{best[0]-acc_m5:+.4f}")

    # 关键：门控 vs 全检索的增益分解（救回/误伤）
    print("\n=== 门控 vs M5 全检索（最优阈值）===")
    tau = best[1]
    # 门控相对全检索：哪些题改判了？
    changed = 0
    for i in ids:
        g_pred = m5[i] if fam[i] < tau else m0[i]
        if g_pred != m5[i]:
            changed += 1
    print(f"  门控改变了 {changed} 题的决策（相对全检索）")

    # 保存结果
    out = []
    for i in ids:
        g_pred = m5[i] if fam[i] < tau else m0[i]
        out.append({
            "id": i,
            "familiarity": fam[i],
            "label": labels[i],
            "M0_pred": m0[i],
            "M5_pred": m5[i],
            "gated_pred": g_pred,
            "gated_correct": g_pred == labels[i],
        })
    with open("results/gating_fam.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"\n已保存 {len(out)} 条到 results/gating_fam.jsonl")


if __name__ == "__main__":
    main()
