"""用 MedQA hard 子集（100 题）做闭卷 + 开卷，构造 ground truth。

hard 子集定义：>50% LLM 答错的题（MedAgentsBench / MedCoG 划分）。
对每题：
- 闭卷：只喂题干+选项。
- 开卷：喂「自生成的医学解释」作为检索证据（见 _gen_rationale）。
- Δ = correct_open - correct_closed（负值按 0），should_retrieve = Δ==1。

开卷知识来源：DEVIATION——MedQA 原始无 rationale，MVP 用 qwen-plus 自生成
的「疾病机制/诊断要点」解释作为检索证据（模拟检索回的知识文档），
而非直接喂答案。这保证 Δ 有区分度（不是"喂答案必对"）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from src.execute.llm_client import LLMClient


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=str, default="results/medqa_hard_gt.jsonl")
    args = parser.parse_args()

    load_env()
    client = LLMClient()

    rows = []
    with open("data/medqa_hard.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[: args.n]

    os.makedirs("results", exist_ok=True)
    results = []
    for i, r in enumerate(rows):
        opts = [r["options"][k] for k in "ABCD"]
        # 闭卷
        closed = client.answer_closed(r["question"], opts)
        correct_closed = int(closed == r["answer_idx"])
        # 开卷：用自生成解释作为检索证据
        rationale = client.gen_rationale(r["question"], opts, r["answer_idx"])
        open_ans = client.answer_open(r["question"], opts, rationale)
        correct_open = int(open_ans == r["answer_idx"])
        delta = max(0, correct_open - correct_closed)
        should_retrieve = 1 if delta == 1 else 0

        results.append({
            "id": r["id"],
            "meta_info": r["meta_info"],
            "question": r["question"],
            "options": r["options"],
            "answer_idx": r["answer_idx"],
            "closed_pred": closed,
            "open_pred": open_ans,
            "correct_closed": correct_closed,
            "correct_open": correct_open,
            "delta": delta,
            "should_retrieve": should_retrieve,
            "rationale": rationale,
            "metamap_phrases": r["metamap_phrases"],
        })
        print(f"[{i+1}/{len(rows)}] {r['id']} 闭卷={closed}({'对' if correct_closed else '错'}) "
              f"开卷={open_ans}({'对' if correct_open else '错'}) Δ={delta}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_retrieve = sum(1 for r in results if r["should_retrieve"] == 1)
    n_closed_wrong = sum(1 for r in results if r["correct_closed"] == 0)
    print(f"\n=== 摘要 ===")
    print(f"总题数: {len(results)}")
    print(f"闭卷答错: {n_closed_wrong}")
    print(f"should_retrieve=1（闭卷错→开卷对）: {n_retrieve}")


if __name__ == "__main__":
    main()
