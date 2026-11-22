"""主实验（真实 MedQA hard 子集版）：在预算 B 下对比各方法。

数据：results/medqa_hard_gt.jsonl（含 100 题 ground truth）。

分类（DEVIATION，如实标注）：
- 类型1（知识缺口型）≈ meta_info == step1（基础医学知识题）
- 类型2（推理难度型）≈ meta_info == step2&3（临床综合推理题）
  原始文档用「罕见病 IDF 高 vs 常见病多条件」切分，但真实 MedQA 无干净 IDF，
  MVP 用官方 step 分层作代理（step1=知识型，step2&3=推理型）。

方法：
- coverage（我们的方法，连续排序）：稀有度×必要性
- M3 Marina（实体流行度）：实体频率低 → 查
- M4 QuCo-RAG（频率+共现）
- M1 MedCoG 自评（模型侧）
- M2 KBM 熟悉度（模型侧）
- random（随机基线）

指标（分题型）：漏检率/误检率/Hit@B/Acc@B
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


def load_gt(path="results/medqa_hard_gt.jsonl") -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_labeled_questions(gt_rows: List[Dict]) -> List[LabeledQuestion]:
    """把 gt 行转成 LabeledQuestion，并做类型1/类型2分类。"""
    qs = []
    for r in gt_rows:
        # 类型映射：step1 -> type1（知识型），step2&3 -> type2（推理型）
        qtype = "type1" if r["meta_info"] == "step1" else "type2"
        # 概念实体：用 metamap_phrases 里的医学短语（MVP 直接用作概念）
        concepts = r["metamap_phrases"][:8]  # 取前 8 个，控制规模
        q = LabeledQuestion(
            question_id=r["id"],
            stem=r["question"],
            options=[r["options"][k] for k in "ABCD"],
            qtype=qtype,
            concepts=concepts,
            answer=r["answer_idx"],
            correct_closed=r["correct_closed"],
            correct_open=r["correct_open"],
            delta=r["delta"],
            should_retrieve=r["should_retrieve"],
        )
        qs.append(q)
    return qs


def build_concept_idf(gt_rows: List[Dict], corpus_rows: List[Dict]) -> Dict[str, float]:
    """用全 MedQA test 集算概念 IDF（频率低=罕见=高缺失）。

    idf(c) = log(N / (1 + df(c)))，N = corpus 题数，df = 含该实体的题数。
    """
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
            key = p.lower()
            idf[key] = math.log(N / (1 + df.get(key, 0)))
    return idf


def main():
    seed = 42
    rng = np.random.default_rng(seed)

    gt_rows = load_gt()
    # 读全语料（用于 IDF）
    corpus_rows = [json.loads(l) for l in open("data/medqa_test.jsonl", encoding="utf-8")]

    questions = build_labeled_questions(gt_rows)
    concept_idf = build_concept_idf(gt_rows, corpus_rows)

    N = len(questions)
    B = max(1, int(0.2 * N))  # 预算 20%
    print(f"N={N}, B={B} (预算 20%)")
    print(f"类型1(step1/知识型): {sum(1 for q in questions if q.qtype=='type1')}")
    print(f"类型2(step2&3/推理型): {sum(1 for q in questions if q.qtype=='type2')}")
    print(f"should_retrieve=1 总数: {sum(1 for q in questions if q.should_retrieve==1)}")

    # 概念共现（QuCo-RAG 用）：MVP 用实体 df 作为共现代理
    from collections import Counter
    df_counter = Counter()
    for r in corpus_rows:
        for p in set(r["metamap_phrases"]):
            df_counter[p.lower()] += 1

    # coverage（我们的方法）
    coverage_sig = CoverageSignal(idf_map=concept_idf, tau=1.0)

    def coverage_score(q):
        # 必要性 ν：MVP 用「是否罕见」作代理——
        # 罕见实体（idf 高）在题→答案路径上更可能"必经"（ν=1），常见实体可绕（ν=0）。
        # DEVIATION: 完整版 ν 由 KG 最短路径算；MVP 用 idf 阈值近似。
        kg_paths = {}
        for c in q.concepts:
            idf_c = concept_idf.get(c.lower(), 0.0)
            kg_paths[c] = {"path"} if idf_c > coverage_sig.idf_median else set()
        return coverage_sig.coverage(q, kg_paths)

    marina = MarinaSignal(idf_map=concept_idf)
    quco = QuCoRAGSignal(idf_map=concept_idf, cooccur=dict(df_counter))

    # M1/M2 模型侧信号：需要重算（自评 + 熟悉度）
    # 但 ground truth 阶段没存这些，这里用 gt 里的 closed 结果做代理
    # DEVIATION: 熟悉度用「闭卷是否答对」代理（答对=熟悉）；自评需额外 API 调用。
    # 为 MVP 先跑通，自评用「是否 should_retrieve」的逆向代理不合法，
    # 这里改为：对每题调 self_eval（真实模型侧信号）。
    print("\n正在计算模型侧信号（自评 + 熟悉度）...")
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
        se = client.self_eval(q.stem, q.options)
        self_eval_scores[q.question_id] = se
        # 熟悉度代理：闭卷答对=熟悉=1，答错=不熟悉=0
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
            decisions = [
                Decision(question_id=q.question_id, action="retrieve" if i in retrieve_ids else "closed", score=0.0)
                for i, q in enumerate(questions)
            ]
        elif mode == "continuous":
            decisions = allocate_top_b(questions, scores, B, threshold=False)
        else:  # binary
            thr = float(np.median(scores)) if scores else 0.5
            decisions = allocate_top_b(questions, scores, B, threshold=True, threshold_value=thr)

        res = evaluate_by_type(questions, decisions)
        all_results[name] = res
        print(f"\n=== {name} ===")
        for qt, r in res.items():
            print(f"  {qt:6s} 漏检率={r['miss_rate']:.2%} 误检率={r['false_alarm']:.2%} "
                  f"Hit@B={r['hit_at_b']:.2%} Acc@B={r['acc_at_b']:.2%} (n={r['n']}, 检索{r['n_retrieve']})")

    # 输出 CSV
    rows = []
    for name, res in all_results.items():
        for qt, r in res.items():
            rows.append({
                "method": name, "qtype": qt,
                "miss_rate": round(r["miss_rate"], 4),
                "false_alarm": round(r["false_alarm"], 4),
                "hit_at_b": round(r["hit_at_b"], 4),
                "acc_at_b": round(r["acc_at_b"], 4),
                "n": r["n"], "n_retrieve": r["n_retrieve"],
            })
    df = pd.DataFrame(rows)
    df.to_csv("results/core_table_medqa.csv", index=False, encoding="utf-8-sig")
    print("\n=== 核心表（已存 results/core_table_medqa.csv）===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
