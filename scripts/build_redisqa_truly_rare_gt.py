"""针对"真罕见病"题（rare_disease 标签低频出现）做闭卷+开卷验证。

目的：验证 H1 假设——模型侧（qwen-plus）是否在"真正罕见"的疾病上"失明"
（闭卷答错），以及喂 ReCOP 的 NORD 证据后能否救回（检索增益 Δ）。

与 build_redisqa_gt.py 的区别：这里筛的是"真罕见"题（标签出现 <=3 次），
而非随机抽样（随机抽样里大量是结核/疟疾等传染病，qwen 闭卷就能答对）。

输出：results/redisqa_truly_rare_gt.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, ".")

from src.execute.llm_client import LLMClient
from scripts.build_redisqa_gt import load_env, load_recop_index, build_evidence


def main():
    load_env()
    client = LLMClient()
    recop_idx = load_recop_index()

    rows = [json.loads(l) for l in open("data/redisqa_truly_rare.jsonl", encoding="utf-8")]
    rng = random.Random(42)
    sample = rng.sample(rows, min(100, len(rows)))
    print(f"真罕见病题总数 {len(rows)}，抽样 {len(sample)} 题验证")

    results = []
    n_pos = 0
    for i, r in enumerate(sample):
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        answer_idx = "ABCD"[r["cop"]]

        closed = client.answer_closed(r["question"], opts)
        correct_closed = int(closed == answer_idx)

        evidence = build_evidence(recop_idx, r["rare_disease"])
        open_ans = client.answer_open(r["question"], opts, evidence)
        correct_open = int(open_ans == answer_idx)

        delta = max(0, correct_open - correct_closed)
        should_retrieve = 1 if delta == 1 else 0
        if should_retrieve:
            n_pos += 1

        results.append({
            "id": f"rare_{i}",
            "question": r["question"],
            "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]},
            "answer_idx": answer_idx,
            "rare_disease": r["rare_disease"],
            "closed_pred": closed,
            "open_pred": open_ans,
            "correct_closed": correct_closed,
            "correct_open": correct_open,
            "delta": delta,
            "should_retrieve": should_retrieve,
            "evidence": evidence,
        })
        print(f"[{i+1}/{len(sample)}] {r['rare_disease'][0]:35s} 闭卷={closed}({'对' if correct_closed else '错'}) "
              f"开卷={open_ans}({'对' if correct_open else '错'}) Δ={delta}")

    with open("results/redisqa_truly_rare_gt.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== 摘要（真罕见病题）===")
    print(f"总题数: {len(results)}")
    print(f"闭卷正确率: {sum(1 for r in results if r['correct_closed']==1)/len(results):.1%}")
    print(f"开卷正确率: {sum(1 for r in results if r['correct_open']==1)/len(results):.1%}")
    print(f"should_retrieve=1（检索能救）: {n_pos}")
    print(f"闭卷答错题数: {sum(1 for r in results if r['correct_closed']==0)}")


if __name__ == "__main__":
    main()
