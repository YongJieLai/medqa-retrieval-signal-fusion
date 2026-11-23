"""用 PrimeKG 真实三元组替换自生成解释，重建 MedQA hard 的 ground truth。

背景（DEVIATION #2 修复）：
- 旧版（medqa_hard_gt.py）：开卷证据 = qwen-plus 自生成解释（同源循环）。
- 新版（本脚本）：开卷证据 = PrimeKG 真实知识三元组（独立于答题模型）。

对每题：
- 闭卷：只喂题干+选项。
- 开卷：喂 PrimeKG 三元组（results/kg_evidence.jsonl 的 evidence 字段）。
- Δ = correct_open - correct_closed，should_retrieve = Δ==1。

对比新旧 Δ 分布，评估"真实证据"是否带来 ground truth 变化。
"""
from __future__ import annotations

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
    load_env()
    client = LLMClient()

    # 读旧 ground truth（含 closed_pred 可直接复用，避免重复调 API）
    old_rows = [json.loads(l) for l in open("results/medqa_hard_gt.jsonl", encoding="utf-8")]
    # 读 KG 证据
    kg_evidence = {json.loads(l)["id"]: json.loads(l) for l in open("results/kg_evidence.jsonl", encoding="utf-8")}

    results = []
    for i, r in enumerate(old_rows):
        qid = r["id"]
        opts = [r["options"][k] for k in "ABCD"]
        # 闭卷复用旧结果（closed 不依赖证据）
        correct_closed = r["correct_closed"]
        closed = r["closed_pred"]

        # 开卷：用 PrimeKG 三元组作为证据
        ev = kg_evidence.get(qid, {})
        evidence = ev.get("evidence", "")
        if evidence:
            open_ans = client.answer_open(r["question"], opts, evidence)
        else:
            # 无证据则开卷=闭卷（无法检索，Δ=0）
            open_ans = closed
        correct_open = int(open_ans == r["answer_idx"])
        delta = max(0, correct_open - correct_closed)
        should_retrieve = 1 if delta == 1 else 0

        results.append({
            "id": qid,
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
            "evidence": evidence,
            "evidence_source": "primekg",
            "anchored_diseases": ev.get("anchored_diseases", []),
            "metamap_phrases": r["metamap_phrases"],
        })
        print(f"[{i+1}/{len(old_rows)}] {qid} 闭卷={closed}({'对' if correct_closed else '错'}) "
              f"开卷={open_ans}({'对' if correct_open else '错'}) Δ={delta} "
              f"证据={('有' if evidence else '无')}")

    with open("results/medqa_hard_gt_primekg.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 摘要 + 新旧对比
    n_retrieve_new = sum(1 for r in results if r["should_retrieve"] == 1)
    n_retrieve_old = sum(1 for r in old_rows if r["should_retrieve"] == 1)
    print(f"\n=== 摘要（PrimeKG 证据）===")
    print(f"should_retrieve=1 题数: 新={n_retrieve_new} 旧={n_retrieve_old}")
    print(f"闭卷答错: {sum(1 for r in results if r['correct_closed']==0)}")
    print(f"开卷答对(新): {sum(1 for r in results if r['correct_open']==1)}")

    # 关键对比：旧 Δ=1 的题，新证据下是否仍 Δ=1
    old_pos_ids = {r["id"] for r in old_rows if r["should_retrieve"] == 1}
    new_pos_ids = {r["id"] for r in results if r["should_retrieve"] == 1}
    print(f"\n=== 新旧 Δ 对比 ===")
    print(f"旧 should_retrieve=1 的题: {len(old_pos_ids)}")
    print(f"新 should_retrieve=1 的题: {len(new_pos_ids)}")
    print(f"交集(旧1→新1): {len(old_pos_ids & new_pos_ids)}")
    print(f"旧1→新0(证据反而让模型不再需要检索?): {len(old_pos_ids - new_pos_ids)}")
    print(f"旧0→新1(新证据让模型从不会到会): {len(new_pos_ids - old_pos_ids)}")


if __name__ == "__main__":
    main()
