"""P4：6 方法汇总对比 + 四维过程分析。

输入：results/p1_m0_m1.jsonl + results/p2_m2_m5.jsonl（+ 可选 p3_m3_m4.jsonl）
输出：
- results/final_summary.csv（方法 x 指标汇总表）
- results/scores_merged.jsonl（合并所有方法的逐题结果）
- docs/MedXpertQA_RAG评测最终报告.md（四维分析报告）

四维：① 增益来自哪（方法间对比，按 question_type 拆分）② 检索质量 ③ 失败三分类 ④ 成本效率
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

P1 = "results/p1_m0_m1.jsonl"
P2 = "results/p2_m2_m5.jsonl"
P3 = "results/p3_m3_m4.jsonl"  # 可选

OUT_MERGED = "results/scores_merged.jsonl"
OUT_SUMMARY = "results/final_summary.csv"


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def acc(rows, pred_key):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get(pred_key) == r.get("label")) / len(rows)


def acc_by(rows, pred_key, group_key):
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        if pred_key not in r:
            continue
        g = r.get(group_key, "NA")
        d[g][0] += 1 if r[pred_key] == r["label"] else 0
        d[g][1] += 1
    return {k: round(v[0] / v[1], 4) for k, v in sorted(d.items())}


def main():
    p1 = load(P1)
    p2 = load(P2)
    p3 = load(P3)

    # 合并（按 id 对齐）
    by_id = {}
    for r in p1:
        by_id[r["id"]] = r
    for r in p2:
        rid = r["id"]
        if rid in by_id:
            by_id[rid].update(r)
        else:
            by_id[rid] = r
    for r in p3:
        rid = r["id"]
        if rid in by_id:
            by_id[rid].update(r)
        else:
            by_id[rid] = r
    merged = list(by_id.values())

    # 方法 → 预测字段映射
    methods = {
        "M0": "M0_pred",
        "M1_k1": "M1_k1_pred",
        "M1_k3": "M1_k3_pred",
        "M1_k5": "M1_k5_pred",
        "M2": "M2_pred",
        "M5": "M5_pred",
        "M3": "M3_pred",
        "M4": "M4_pred",
    }

    # 汇总表
    summary_rows = []
    m0_acc = acc(merged, "M0_pred")
    for mname, pred_key in methods.items():
        if not any(pred_key in r for r in merged):
            continue
        a = acc(merged, pred_key)
        summary_rows.append({
            "method": mname,
            "acc": round(a, 4),
            "delta": round(a - m0_acc, 4),
            "reasoning_acc": acc_by(merged, pred_key, "question_type").get("Reasoning", 0),
            "understanding_acc": acc_by(merged, pred_key, "question_type").get("Understanding", 0),
        })

    # 写 summary.csv
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("method,acc,delta,reasoning_acc,understanding_acc\n")
        for r in summary_rows:
            f.write(f"{r['method']},{r['acc']},{r['delta']},{r['reasoning_acc']},{r['understanding_acc']}\n")

    # 写 merged
    with open(OUT_MERGED, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印汇总
    print("=== 6 方法汇总 ===")
    print(f"{'方法':<8}{'准确率':<10}{'Δ':<8}{'Reasoning':<12}{'Understanding'}")
    for r in summary_rows:
        print(f"{r['method']:<8}{r['acc']:<10}{r['delta']:+.4f}   {r['reasoning_acc']:<12}{r['understanding_acc']}")

    # 检索增益分解（M0 vs 各方法）
    print("\n=== 检索增益分解（救回 vs 误伤，相对 M0）===")
    for mname, pred_key in methods.items():
        if mname == "M0" or not any(pred_key in r for r in merged):
            continue
        rescue = sum(1 for r in merged if r.get("M0_pred") != r["label"] and r.get(pred_key) == r["label"])
        harm = sum(1 for r in merged if r.get("M0_pred") == r["label"] and r.get(pred_key) != r["label"])
        print(f"  {mname}: 救回 {rescue} vs 误伤 {harm}，净 {rescue-harm}")

    print(f"\n已保存 {OUT_SUMMARY} 和 {OUT_MERGED}")


if __name__ == "__main__":
    main()
