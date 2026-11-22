"""主实验（混合批次版）：混合难易题的场景验证。

数据：results/mixed_batch_gt.jsonl（50 题，难:易=25:25）。

这是任务交接文档要的"混合长尾批次"——难样本（罕见/该检索）+ 易样本（不该检索），
预算 B=20% 稀缺，检验各方法能否精准挑出"该检索"的题。

分类（DEVIATION）：类型1=step1（知识型），类型2=step2&3（推理型）。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.signal.coverage import CoverageSignal
from src.signal.baselines import MarinaSignal, QuCoRAGSignal, MedCoGSelfEvalSignal, KBMFamiliaritySignal
from src.allocation.allocator import allocate_top_b
from src.eval.metrics import evaluate_by_type
from src.schema import LabeledQuestion, Decision


def load_gt(path="results/mixed_batch_gt.jsonl") -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_labeled_questions(gt_rows):
    qs = []
    for r in gt_rows:
        qtype = "type1" if r["meta_info"] == "step1" else "type2"
        q = LabeledQuestion(
            question_id=r["id"],
            stem=r["question"],
            options=[r["options"][k] for k in "ABCD"],
            qtype=qtype,
            concepts=r["metamap_phrases"][:8],
            answer=r["answer_idx"],
            correct_closed=r["correct_closed"],
            correct_open=r["correct_open"],
            delta=r["delta"],
            should_retrieve=r["should_retrieve"],
        )
        qs.append(q)
    return qs


def build_concept_idf(gt_rows, corpus_rows):
    import math
    from collections import Counter
    N = len(corpus_rows)
    df = Counter()
    for r in corpus_rows:
        for p in set(r["metamap_phrases"]):
            df[p.lower()] += 1
    idf = {}
    for r in gt_rows:
        for p in r["metamap_phrases"]:
            idf[p.lower()] = math.log(N / (1 + df.get(p.lower(), 0)))
    return idf


def main():
    seed = 42
    rng = np.random.default_rng(seed)

    gt_rows = load_gt()
    corpus_rows = [json.loads(l) for l in open("data/medqa_test.jsonl", encoding="utf-8")]
    questions = build_labeled_questions(gt_rows)
    concept_idf = build_concept_idf(gt_rows, corpus_rows)

    N = len(questions)
    B = max(1, int(0.2 * N))
    print(f"N={N}, B={B} (预算 20%)")
    print(f"类型1(step1/知识型): {sum(1 for q in questions if q.qtype=='type1')}")
    print(f"类型2(step2&3/推理型): {sum(1 for q in questions if q.qtype=='type2')}")
    print(f"should_retrieve=1: {sum(1 for q in questions if q.should_retrieve==1)}")

    from collections import Counter
    df_counter = Counter()
    for r in corpus_rows:
        for p in set(r["metamap_phrases"]):
            df_counter[p.lower()] += 1

    coverage_sig = CoverageSignal(idf_map=concept_idf, tau=1.0)

    def coverage_score(q):
        kg_paths = {}
        for c in q.concepts:
            idf_c = concept_idf.get(c.lower(), 0.0)
            kg_paths[c] = {"path"} if idf_c > coverage_sig.idf_median else set()
        return coverage_sig.coverage(q, kg_paths)

    marina = MarinaSignal(idf_map=concept_idf)
    quco = QuCoRAGSignal(idf_map=concept_idf, cooccur=dict(df_counter))

    print("\n计算模型侧信号（自评 + 熟悉度）...")
    from src.execute.llm_client import LLMClient
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    client = LLMClient()
    self_eval_scores = {}
    familiarity = {}
    for q in questions:
        self_eval_scores[q.question_id] = client.self_eval(q.stem, q.options)
        familiarity[q.question_id] = float(q.correct_closed)
    self_eval_sig = MedCoGSelfEvalSignal(self_eval_scores)
    kbm_sig = KBMFamiliaritySignal(familiarity)

    methods = {
        "coverage(ours)": ("continuous", coverage_score),
        "M3_Marina": ("binary", lambda q: marina.score(q)),
        "M4_QuCo-RAG": ("binary", lambda q: quco.score(q)),
        "M1_MedCoG自评": ("continuous", lambda q: self_eval_sig.score(q)),
        "M2_KBM熟悉度": ("continuous", lambda q: kbm_sig.score(q)),
        "random": ("random", lambda q: 0.0),
    }

    all_results = {}
    for name, (mode, score_fn) in methods.items():
        scores = [score_fn(q) for q in questions]
        if mode == "random":
            perm = rng.permutation(N)
            retrieve_ids = set(perm[:B].tolist())
            decisions = [Decision(question_id=q.question_id, action="retrieve" if i in retrieve_ids else "closed", score=0.0) for i, q in enumerate(questions)]
        elif mode == "continuous":
            decisions = allocate_top_b(questions, scores, B, threshold=False)
        else:
            thr = float(np.median(scores)) if scores else 0.5
            decisions = allocate_top_b(questions, scores, B, threshold=True, threshold_value=thr)
        res = evaluate_by_type(questions, decisions)
        all_results[name] = res
        print(f"\n=== {name} ===")
        for qt, r in res.items():
            print(f"  {qt:6s} 漏检率={r['miss_rate']:.2%} 误检率={r['false_alarm']:.2%} Hit@B={r['hit_at_b']:.2%} Acc@B={r['acc_at_b']:.2%} (n={r['n']}, 检索{r['n_retrieve']})")

    rows = []
    for name, res in all_results.items():
        for qt, r in res.items():
            rows.append({"method": name, "qtype": qt, "miss_rate": round(r["miss_rate"],4), "false_alarm": round(r["false_alarm"],4), "hit_at_b": round(r["hit_at_b"],4), "acc_at_b": round(r["acc_at_b"],4), "n": r["n"], "n_retrieve": r["n_retrieve"]})
    df = pd.DataFrame(rows)
    df.to_csv("results/core_table_mixed.csv", index=False, encoding="utf-8-sig")
    print("\n=== 核心表（已存 results/core_table_mixed.csv）===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
