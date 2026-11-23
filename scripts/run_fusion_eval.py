"""融合互补性评测：知识侧 × 模型侧 → 融合 > 单独。

主叙事（H1 修正后）：知识侧(coverage)与模型侧(自评/熟悉度)覆盖不同错误类型，
融合后漏检率应低于任一单独信号。

方法：
  - coverage（知识侧，ours）
  - M1 self_eval（模型侧自评）
  - M2 familiarity（模型侧熟悉度，1-familiarity 越高越该查）
  - fusion_rank_max（coverage × self_eval）
  - fusion_rank_max（coverage × familiarity）
  - fusion_rank_sum（coverage × self_eval，对照）

指标：漏检率（越低越好）/ 误检率 / Hit@B / Acc@B。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.allocation.allocator import allocate_top_b
from src.eval.metrics import evaluate, evaluate_by_type
from src.schema import LabeledQuestion, Decision
from src.signal.fusion import fuse


def load_gt():
    return [json.loads(l) for l in open("results/mixed_batch_gt.jsonl", encoding="utf-8")]


def build_labeled(gt_rows):
    qs = []
    for r in gt_rows:
        qtype = "type1" if r["meta_info"] == "step1" else "type2"
        q = LabeledQuestion(
            question_id=r["id"], stem=r["question"],
            options=[r["options"][k] for k in "ABCD"], qtype=qtype,
            concepts=r["metamap_phrases"][:8], answer=r["answer_idx"],
            correct_closed=r["correct_closed"], correct_open=r["correct_open"],
            delta=r["delta"], should_retrieve=r["should_retrieve"],
        )
        qs.append(q)
    return qs


def main():
    gt_rows = load_gt()
    questions = build_labeled(gt_rows)
    N = len(questions)
    B = max(1, int(0.2 * N))

    cov_scores = json.load(open("results/coverage_scores.json", encoding="utf-8"))
    model_sig = json.load(open("results/model_signals_real.json", encoding="utf-8"))

    # 信号向量（对齐 questions 顺序）
    cov = [cov_scores.get(q.question_id, 0.0) for q in questions]
    self_eval = [model_sig.get(q.question_id, {}).get("self_eval", 0.5) for q in questions]
    familiarity = [1.0 - model_sig.get(q.question_id, {}).get("familiarity", 0.5) for q in questions]

    methods = {
        "coverage(知识侧)": cov,
        "M1_self_eval(模型侧)": self_eval,
        "M2_familiarity(模型侧)": familiarity,
        "融合_cov×self_eval(rank_max)": fuse(questions, cov, self_eval, "rank_max"),
        "融合_cov×fam(rank_max)": fuse(questions, cov, familiarity, "rank_max"),
        "融合_cov×self_eval(rank_sum)": fuse(questions, cov, self_eval, "rank_sum"),
    }

    print(f"N={N}, B={B}")
    print(f"type1={sum(1 for q in questions if q.qtype=='type1')}, "
          f"type2={sum(1 for q in questions if q.qtype=='type2')}")
    print(f"should_retrieve=1: {sum(1 for q in questions if q.should_retrieve==1)}\n")

    all_results = {}
    for name, scores in methods.items():
        decisions = allocate_top_b(questions, scores, B, threshold=False)
        res = evaluate_by_type(questions, decisions)
        all_results[name] = res
        print(f"=== {name} ===")
        for qt, r in res.items():
            print(f"  {qt:6s} 漏检率={r['miss_rate']:.2%} 误检率={r['false_alarm']:.2%} "
                  f"Hit@B={r['hit_at_b']:.2%} Acc@B={r['acc_at_b']:.2%} "
                  f"(n={r['n']},检索{r['n_retrieve']})")

    # 落表
    rows = []
    for name, res in all_results.items():
        for qt, r in res.items():
            rows.append({"method": name, "qtype": qt,
                         "miss_rate": round(r["miss_rate"], 4),
                         "false_alarm": round(r["false_alarm"], 4),
                         "hit_at_b": round(r["hit_at_b"], 4),
                         "acc_at_b": round(r["acc_at_b"], 4),
                         "n": r["n"], "n_retrieve": r["n_retrieve"]})
    df = pd.DataFrame(rows)
    df.to_csv("results/core_table_fusion.csv", index=False, encoding="utf-8-sig")
    print("\n=== 融合核心表（results/core_table_fusion.csv）===")
    print(df.to_string(index=False))

    # 关键结论：融合 vs 单独，整体漏检率对比
    print("\n=== 互补性检验（整体漏检率，越低越好）===")
    base_miss = all_results["coverage(知识侧)"]["all"]["miss_rate"]
    model_miss = all_results["M1_self_eval(模型侧)"]["all"]["miss_rate"]
    fused_miss = all_results["融合_cov×self_eval(rank_max)"]["all"]["miss_rate"]
    print(f"  知识侧漏检率      : {base_miss:.2%}")
    print(f"  模型侧漏检率      : {model_miss:.2%}")
    print(f"  融合漏检率        : {fused_miss:.2%}")
    if fused_miss < min(base_miss, model_miss):
        print("  ✅ 融合漏检率 < 任一单独信号（互补性成立）")
    elif fused_miss < base_miss and fused_miss < model_miss:
        print("  ✅ 融合漏检率低于两者（互补性成立）")
    else:
        print("  ⚠️ 融合未优于单独信号，需检查信号质量/融合方式")


if __name__ == "__main__":
    main()
