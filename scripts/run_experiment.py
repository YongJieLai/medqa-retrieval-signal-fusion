"""主实验：在预算 B 下对比各方法，输出核心表。

方法：
- coverage（我们的方法，连续排序）
- M3 Marina（实体流行度，二元阈值）
- M4 QuCo-RAG（频率+共现，二元阈值）
- M1 MedCoG 自评（模型侧，连续排序）
- M2 KBM 熟悉度（模型侧，连续排序）
- random（随机基线，用于 H3 对照）

指标（分题型 type1/type2 + 整体）：
漏检率 / 误检率 / Hit@B / Acc@B

读取 results/ground_truth.json + results/llm_signals.json（由 build_ground_truth.py 生成）。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data.synthetic_batch import build_synthetic_batch, get_concept_idf
from src.signal.coverage import CoverageSignal
from src.signal.baselines import MarinaSignal, QuCoRAGSignal, MedCoGSelfEvalSignal, KBMFamiliaritySignal
from src.allocation.allocator import allocate_top_b
from src.eval.metrics import evaluate_by_type


def load_ground_truth() -> List[Dict]:
    with open("results/ground_truth.json", encoding="utf-8") as f:
        return json.load(f)


def load_signals() -> Dict:
    with open("results/llm_signals.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    seed = 42
    rng = np.random.default_rng(seed)

    gt_rows = load_ground_truth()
    signals = load_signals()
    questions = build_synthetic_batch()

    # 用 gt 填充 labeled question
    gt_by_id = {r["question_id"]: r for r in gt_rows}
    for q in questions:
        r = gt_by_id[q.question_id]
        q.correct_closed = r["correct_closed"]
        q.correct_open = r["correct_open"]
        q.delta = r["delta"]
        q.should_retrieve = r["should_retrieve"]

    N = len(questions)
    B = max(1, int(0.2 * N))
    print(f"N={N}, B={B} (预算 20%)")

    concept_idf = get_concept_idf()

    # 构造各方法的评分器
    # coverage（我们的方法）：需要「概念是否在题→答案唯一路径」的 MVP 判断。
    # MVP 简化：类型1（罕见病）的疾病/药物概念是「必经节点」（ν=1），
    # 类型2（推理题）的常见病概念可绕（ν=0），因为知识模型已有、不是缺口。
    # DEVIATION: 完整版 ν_c 由 KG 最短路径结构计算；MVP 用类型先验近似：
    #   类型1 的核心罕见概念 ν=1（唯一路径、无可绕），类型2 的常见概念 ν=0（可绕/已掌握）。
    coverage_sig = CoverageSignal(idf_map=concept_idf, tau=1.0)

    # 各方法给每道题打连续分数
    def coverage_score(q):
        # MVP 必要性：罕见病题（type1）核心概念 ν=1，推理题（type2）概念 ν=0
        kg_paths = {}
        for c in q.concepts:
            if q.qtype == "type1":
                kg_paths[c] = {"path"}  # 罕见概念必经
            else:
                kg_paths[c] = set()  # 常见概念可绕/已掌握
        return coverage_sig.coverage(q, kg_paths)

    marina = MarinaSignal(idf_map=concept_idf)
    quco = QuCoRAGSignal(idf_map=concept_idf, cooccur={c: 1 for c in concept_idf})  # MVP：无真实共现，全给 1
    self_eval = MedCoGSelfEvalSignal({qid: s["self_eval"] for qid, s in signals.items()})
    kbm = KBMFamiliaritySignal({qid: s["familiarity"] for qid, s in signals.items()})

    methods = {
        "coverage(ours)": ("continuous", coverage_score),
        "M3_Marina": ("binary", lambda q: marina.score(q)),
        "M4_QuCo-RAG": ("binary", lambda q: quco.score(q)),
        "M1_MedCoG自评": ("continuous", lambda q: self_eval.score(q)),
        "M2_KBM熟悉度": ("continuous", lambda q: kbm.score(q)),
        "random": ("random", lambda q: 0.0),
    }

    # 计算各方法分数
    all_results = {}
    for name, (mode, score_fn) in methods.items():
        scores = [score_fn(q) for q in questions]

        if mode == "random":
            # 随机基线：随机选 B 道
            perm = rng.permutation(N)
            decisions = allocate_top_b(questions, [0.0] * N, B)
            # 手动打乱顺序实现随机
            from src.schema import Decision
            retrieve_ids = set(perm[:B].tolist())
            decisions = [
                Decision(question_id=q.question_id, action="retrieve" if i in retrieve_ids else "closed", score=0.0)
                for i, q in enumerate(questions)
            ]
        elif mode == "continuous":
            decisions = allocate_top_b(questions, scores, B, threshold=False)
        else:  # binary：阈值方法
            # 阈值取分数中位数（MVP 简化，等价于「高频/低频」切分）
            thr = float(np.median(scores)) if scores else 0.5
            decisions = allocate_top_b(questions, scores, B, threshold=True, threshold_value=thr)

        res = evaluate_by_type(questions, decisions)
        all_results[name] = res
        print(f"\n=== {name} ===")
        for qt, r in res.items():
            print(
                f"  {qt:6s} 漏检率={r['miss_rate']:.2%} 误检率={r['false_alarm']:.2%} "
                f"Hit@B={r['hit_at_b']:.2%} Acc@B={r['acc_at_b']:.2%} (n={r['n']}, 检索{r['n_retrieve']})"
            )

    # 输出核心表 CSV
    rows = []
    for name, res in all_results.items():
        for qt, r in res.items():
            rows.append(
                {
                    "method": name,
                    "qtype": qt,
                    "miss_rate": round(r["miss_rate"], 4),
                    "false_alarm": round(r["false_alarm"], 4),
                    "hit_at_b": round(r["hit_at_b"], 4),
                    "acc_at_b": round(r["acc_at_b"], 4),
                    "n": r["n"],
                    "n_retrieve": r["n_retrieve"],
                }
            )
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/core_table.csv", index=False, encoding="utf-8-sig")
    print("\n=== 核心表（已存 results/core_table.csv）===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
