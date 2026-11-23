"""用 ReDis-QA（罕见病题）+ ReCOP（NORD 证据）构造干净 ground truth。

这是对之前 MedQA hard ground truth 的替换（修复两个问题）：
1. 罕见病题集：ReDis-QA 是真正罕见病题（带 rare disease 标注），
   而非 MedQA hard 的"难题集合"（伦理/决策题）。
2. 开卷证据：ReCOP 的 NORD 权威知识（独立于答题模型），
   而非 qwen-plus 自生成解释（旧版严重泄漏答案，开卷正确率 98%）。

对每题：
- 闭卷：只喂题干 + 4 选项。
- 开卷：喂该罕见病在 ReCOP 里的 Overview/Symptoms/Causes/Diagnosis 知识。
- Δ = correct_open - correct_closed，should_retrieve = Δ==1。

证据选择纪律：不用 Therapies（治疗）和 Affects/Related disorders，
避免证据直接包含"该病的标准治疗/特征"从而过度泄漏答案；
用 Overview/Symptoms/Causes/Diagnosis（临床描述性知识），
让"开卷答对"真正体现"补足了模型缺失的背景知识"。

输出：results/redisqa_gt.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, ".")

from src.execute.llm_client import LLMClient

# 用作开卷证据的 ReCOP 类别（按泄漏风险从低到高排序）
EVIDENCE_TITLES = ["Overview", "Symptoms", "Causes", "Diagnosis"]
MAX_EVIDENCE_CHARS = 2000  # 证据截断，避免过长


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_recop_index() -> Dict[str, Dict[str, str]]:
    """构建 ReCOP 索引：{罕见病名: {title类别: content}}"""
    idx = defaultdict(dict)
    for line in open("data/recop.jsonl", encoding="utf-8"):
        r = json.loads(line)
        disease = r["rare_disease"]
        # title 形如 "Overview of 47, XXY (Klinefelter Syndrome)"，取 " of " 前的类别
        cat = r["title"].split(" of ")[0] if " of " in r["title"] else r["title"]
        idx[disease][cat] = r["content"]
    return idx


def build_evidence(recop_idx: Dict[str, Dict[str, str]], diseases: List[str]) -> str:
    """按疾病 + 类别拼接开卷证据。"""
    parts = []
    for d in diseases:
        if d not in recop_idx:
            continue
        for cat in EVIDENCE_TITLES:
            if cat in recop_idx[d]:
                parts.append(f"【{d} · {cat}】{recop_idx[d][cat]}")
    evidence = "\n\n".join(parts)
    return evidence[:MAX_EVIDENCE_CHARS]


def main():
    load_env()
    client = LLMClient()
    recop_idx = load_recop_index()
    print(f"ReCOP 索引疾病数: {len(recop_idx)}")

    rows = [json.loads(l) for l in open("data/redisqa.jsonl", encoding="utf-8")]
    # 最小验证：抽样 100 题（确保覆盖多种罕见病）
    import random
    rng = random.Random(42)
    sample = rng.sample(rows, 100)

    results = []
    n_pos = 0
    n_closed_wrong = 0
    for i, r in enumerate(sample):
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        answer_idx = "ABCD"[r["cop"]]  # cop 是 0-based

        # 闭卷
        closed = client.answer_closed(r["question"], opts)
        correct_closed = int(closed == answer_idx)

        # 开卷：喂 ReCOP 证据
        evidence = build_evidence(recop_idx, r["rare_disease"])
        open_ans = client.answer_open(r["question"], opts, evidence)
        correct_open = int(open_ans == answer_idx)

        delta = max(0, correct_open - correct_closed)
        should_retrieve = 1 if delta == 1 else 0
        if should_retrieve:
            n_pos += 1
        if correct_closed == 0:
            n_closed_wrong += 1

        results.append({
            "id": f"redisqa_{i}",
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
        print(f"[{i+1}/{len(sample)}] {r['rare_disease'][0]} 闭卷={closed}({'对' if correct_closed else '错'}) "
              f"开卷={open_ans}({'对' if correct_open else '错'}) Δ={delta}")

    with open("results/redisqa_gt.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== 摘要 ===")
    print(f"总题数: {len(results)}")
    print(f"闭卷正确率: {sum(1 for r in results if r['correct_closed']==1)/len(results):.1%}")
    print(f"开卷正确率: {sum(1 for r in results if r['correct_open']==1)/len(results):.1%}")
    print(f"should_retrieve=1（闭卷错→开卷对）: {n_pos}")
    print(f"闭卷答错题数: {n_closed_wrong}")


if __name__ == "__main__":
    main()
