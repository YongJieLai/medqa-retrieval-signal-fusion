"""P4 语义检索版对比分析：BM25 vs MedCPT 语义检索，六方法横向对比。

汇总：
- P1 BM25 版（results/p1_m0_m1.jsonl）
- P2 BM25 版（results/p2_m2_m5.jsonl）
- P1 语义版（results/p1_semantic.jsonl）
- P2 语义版（results/p2_semantic.jsonl）

产出：
1. 准确率对比表（BM25 vs 语义，六方法）
2. 检索增益分解（救回 vs 误伤）
3. 按 question_type / medical_task 分组
4. 检索质量对比（top1 分数分布、标题相关性）
"""
from __future__ import annotations

import json
from collections import Counter


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_dict(path, key="id"):
    return {json.loads(l)[key]: json.loads(l) for l in open(path, encoding="utf-8")}


def acc(rows, pred_key):
    n = len(rows)
    c = sum(1 for r in rows if r[pred_key] == r["label"])
    return round(c / n, 4)


def acc_by(rows, pred_key, group_key):
    d = {}
    for r in rows:
        d.setdefault(r[group_key], [0, 0])
        d[r[group_key]][0] += 1 if r[pred_key] == r["label"] else 0
        d[r[group_key]][1] += 1
    return {k: round(v[0] / v[1], 4) for k, v in sorted(d.items())}


def main():
    p1_bm = load("results/p1_m0_m1.jsonl")
    p2_bm = load("results/p2_m2_m5.jsonl")
    p1_sem = load("results/p1_semantic.jsonl")
    p2_sem = load("results/p2_semantic.jsonl")

    # 合并（BM25 版里 M0/M1，语义版里 M0 复用 + M1 语义）
    m0 = {r["id"]: r["M0_pred"] for r in p1_bm}  # M0 基线（BM25 和语义共用）

    print("=" * 70)
    print("六方法准确率对比：BM25 vs MedCPT 语义检索")
    print("=" * 70)

    # 表格
    methods = [
        ("M1 k=1", {r["id"]: r["M1_k1_pred"] for r in p1_bm},
         {r["id"]: r["M1_k1_pred"] for r in p1_sem}),
        ("M1 k=3", {r["id"]: r["M1_k3_pred"] for r in p1_bm},
         {r["id"]: r["M1_k3_pred"] for r in p1_sem}),
        ("M1 k=5", {r["id"]: r["M1_k5_pred"] for r in p1_bm},
         {r["id"]: r["M1_k5_pred"] for r in p1_sem}),
        ("M2 推理查询扩展", {r["id"]: r["M2_pred"] for r in p2_bm},
         {r["id"]: r["M2_pred"] for r in p2_sem}),
        ("M5 证据过滤", {r["id"]: r["M5_pred"] for r in p2_bm},
         {r["id"]: r["M5_pred"] for r in p2_sem}),
    ]

    labels = {r["id"]: r["label"] for r in p1_bm}
    ids = list(labels.keys())

    print(f"\n{'方法':<20} {'BM25':>8} {'语义':>8} {'Δ':>8}")
    print("-" * 50)
    m0_acc = sum(1 for i in ids if m0[i] == labels[i]) / len(ids)
    print(f"{'M0 闭卷基线':<20} {m0_acc:>8.4f} {'—':>8} {'—':>8}")
    for name, bm, sem in methods:
        c_bm = sum(1 for i in ids if bm[i] == labels[i])
        c_sem = sum(1 for i in ids if sem[i] == labels[i])
        a_bm = c_bm / len(ids)
        a_sem = c_sem / len(ids)
        print(f"{name:<20} {a_bm:>8.4f} {a_sem:>8.4f} {a_sem - a_bm:>+8.4f}")

    # 检索增益分解（语义 vs BM25 相对 M0）
    print("\n" + "=" * 70)
    print("检索增益分解（相对 M0 闭卷）：救回 vs 误伤")
    print("=" * 70)
    for name, bm, sem in methods:
        for label_src, preds in [("BM25", bm), ("语义", sem)]:
            rescue = sum(1 for i in ids if m0[i] != labels[i] and preds[i] == labels[i])
            harm = sum(1 for i in ids if m0[i] == labels[i] and preds[i] != labels[i])
            print(f"  {name:<20} {label_src}: 救回{rescue:3d} vs 误伤{harm:3d} 净{rescue-harm:+d}")

    # 核心假设：M2 vs M1_k3（语义版）
    print("\n" + "=" * 70)
    print("核心假设验证（语义检索版）：M2 推理先行 vs M1_k3 先检索")
    print("=" * 70)
    m1_k3_sem = {r["id"]: r["M1_k3_pred"] for r in p1_sem}
    m2_sem = {r["id"]: r["M2_pred"] for r in p2_sem}
    rescue = sum(1 for i in ids if m1_k3_sem[i] != labels[i] and m2_sem[i] == labels[i])
    harm = sum(1 for i in ids if m1_k3_sem[i] == labels[i] and m2_sem[i] != labels[i])
    print(f"  M2 vs M1_k3（语义）: 救回{rescue} vs 误伤{harm} 净{rescue-harm:+d}")

    # 分组分析（语义版）
    print("\n" + "=" * 70)
    print("语义检索版按 question_type 分组")
    print("=" * 70)
    p1_sem_dict = {r["id"]: r for r in p1_sem}
    p2_sem_dict = {r["id"]: r for r in p2_sem}
    # 构造合并行
    merged = []
    for i in ids:
        r1 = p1_sem_dict[i]
        r2 = p2_sem_dict[i]
        merged.append({
            "id": i,
            "label": labels[i],
            "question_type": r1["question_type"],
            "medical_task": r1["medical_task"],
            "M0_pred": m0[i],
            "M1_k3_pred": r1["M1_k3_pred"],
            "M2_pred": r2["M2_pred"],
            "M5_pred": r2["M5_pred"],
        })

    for qt in ["Reasoning", "Understanding"]:
        sub = [r for r in merged if r["question_type"] == qt]
        print(f"\n  {qt}（n={len(sub)}）:")
        for pred in ["M0_pred", "M1_k3_pred", "M2_pred", "M5_pred"]:
            a = acc(sub, pred)
            print(f"    {pred:<12}: {a:.4f}")

    for task in ["Diagnosis", "Treatment", "Basic Science"]:
        sub = [r for r in merged if r["medical_task"] == task]
        if sub:
            print(f"\n  {task}（n={len(sub)}）:")
            for pred in ["M0_pred", "M1_k3_pred", "M2_pred", "M5_pred"]:
                a = acc(sub, pred)
                print(f"    {pred:<12}: {a:.4f}")

    # 检索质量：语义 top1 分数分布
    print("\n" + "=" * 70)
    print("语义检索质量：top1 内积分分布")
    print("=" * 70)
    sem_all = load_dict("results/semantic_retrieval.jsonl")
    scores = [sem_all[i]["raw_query"][0]["score"] for i in ids]
    import statistics
    print(f"  raw_query top1 分数: mean={statistics.mean(scores):.3f} min={min(scores):.3f} max={max(scores):.3f}")

    print("\n分析完成。")


if __name__ == "__main__":
    main()
