"""最终主实验：混合批次 + 真实信号（轻量共现图 coverage + logprobs 模型信号）。

方法：
- coverage（我们的方法，轻量共现图算必要性 ν）
- M1 MedCoG 自评（模型自评分数）
- M2 KBM 熟悉度（logprobs margin，真实内部信号）
- M3 Marina（实体 IDF）
- M4 QuCo-RAG（频率+共现）
- random

指标：漏检率/误检率/Hit@B/Acc@B，分类型1(step1)/类型2(step2&3)。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.allocation.allocator import allocate_top_b
from src.eval.metrics import evaluate_by_type
from src.schema import LabeledQuestion, Decision


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
    seed = 42
    rng = np.random.default_rng(seed)
    gt_rows = load_gt()
    questions = build_labeled(gt_rows)

    # 加载预计算的 coverage 分数（轻量共现图）
    cov_scores = json.load(open("results/coverage_scores.json", encoding="utf-8"))
    # 加载真实模型信号
    model_sig = json.load(open("results/model_signals_real.json", encoding="utf-8"))

    # Marina / QuCo 用 IDF（从 medqa_test 算）
    import math
    from collections import Counter
    corpus = [json.loads(l) for l in open("data/medqa_test.jsonl", encoding="utf-8")]
    N_corpus = len(corpus)
    df = Counter()
    for r in corpus:
        for p in set(r["metamap_phrases"]):
            df[p.lower()] += 1
    def idf_of(phrases):
        return max([math.log(N_corpus/(1+df.get(p.lower(),0))) for p in phrases] or [0.0])

    N = len(questions)
    B = max(1, int(0.2 * N))
    print(f"N={N}, B={B}")
    print(f"类型1(step1): {sum(1 for q in questions if q.qtype=='type1')}, 类型2(step2&3): {sum(1 for q in questions if q.qtype=='type2')}")
    print(f"should_retrieve=1: {sum(1 for q in questions if q.should_retrieve==1)}")

    methods = {
        "coverage(ours)": ("continuous", lambda q: cov_scores.get(q.question_id, 0.0)),
        "M1_MedCoG自评": ("continuous", lambda q: model_sig.get(q.question_id, {}).get("self_eval", 0.5)),
        # M2 熟悉度：低熟悉度 → 判查，所以分数 = 1 - familiarity
        "M2_KBM熟悉度": ("continuous", lambda q: 1.0 - model_sig.get(q.question_id, {}).get("familiarity", 0.5)),
        "M3_Marina": ("binary", lambda q: idf_of(q.concepts)),
        "M4_QuCo-RAG": ("binary", lambda q: idf_of(q.concepts)),
        "random": ("random", lambda q: 0.0),
    }

    all_results = {}
    for name, (mode, fn) in methods.items():
        scores = [fn(q) for q in questions]
        if mode == "random":
            perm = rng.permutation(N)
            ids = set(perm[:B].tolist())
            decisions = [Decision(question_id=q.question_id, action="retrieve" if i in ids else "closed", score=0.0) for i, q in enumerate(questions)]
        elif mode == "continuous":
            decisions = allocate_top_b(questions, scores, B, threshold=False)
        else:
            thr = float(np.median(scores)) if scores else 0.5
            decisions = allocate_top_b(questions, scores, B, threshold=True, threshold_value=thr)
        res = evaluate_by_type(questions, decisions)
        all_results[name] = res
        print(f"\n=== {name} ===")
        for qt, r in res.items():
            print(f"  {qt:6s} 漏检率={r['miss_rate']:.2%} 误检率={r['false_alarm']:.2%} Hit@B={r['hit_at_b']:.2%} Acc@B={r['acc_at_b']:.2%} (n={r['n']},检索{r['n_retrieve']})")

    rows = []
    for name, res in all_results.items():
        for qt, r in res.items():
            rows.append({"method": name, "qtype": qt, "miss_rate": round(r["miss_rate"],4), "false_alarm": round(r["false_alarm"],4), "hit_at_b": round(r["hit_at_b"],4), "acc_at_b": round(r["acc_at_b"],4), "n": r["n"], "n_retrieve": r["n_retrieve"]})
    df = pd.DataFrame(rows)
    df.to_csv("results/core_table_final.csv", index=False, encoding="utf-8-sig")
    print("\n=== 最终核心表（results/core_table_final.csv）===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
