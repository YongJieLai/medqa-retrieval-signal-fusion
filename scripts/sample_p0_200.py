"""P0 失效基线：抽样 200 题 + 构造去选项检索 query。

按清单 §1.3 分层抽样（question_type × medical_task，random_state=42），
产出 medxpertqa_p0_200.jsonl；按 §1.4 构造去选项检索 query。
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import pandas as pd

DATA = "data/medxpertqa_text.jsonl"
OUT = "data/medxpertqa_p0_200.jsonl"

# §1.3 目标数量（question_type × medical_task）
TARGET = {
    ("Reasoning", "Diagnosis"): 66,
    ("Reasoning", "Treatment"): 46,
    ("Reasoning", "Basic Science"): 40,
    ("Understanding", "Diagnosis"): 20,
    ("Understanding", "Treatment"): 15,
    ("Understanding", "Basic Science"): 13,
}


def get_query(question: str) -> str:
    """§1.4 检索 query 只用题干主体，切掉 Answer Choices 及之后。"""
    return question.split("Answer Choices:")[0].strip()


def main():
    df = pd.read_json(DATA, lines=True)

    sampled = []
    for (qt, mt), n in TARGET.items():
        g = df[(df["question_type"] == qt) & (df["medical_task"] == mt)]
        if len(g) >= n:
            sub = g.sample(n=n, random_state=42)
        else:
            sub = g  # 不足则全取
            print(f"  [warn] {qt}/{mt} 只有 {len(g)} 题，目标 {n}")
        sampled.append(sub)
    sample = pd.concat(sampled, ignore_index=True)

    # 构造去选项 query
    sample["query"] = sample["question"].apply(get_query)

    sample.to_json(OUT, orient="records", lines=True, force_ascii=False)

    print(f"抽样完成: {len(sample)} 题 -> {OUT}")
    print("分层分布:")
    for (qt, mt), c in sample.groupby(["question_type", "medical_task"]).size().items():
        print(f"  {qt}/{mt}: {c}")
    print(f"\nquery 长度分布: 均值={sample['query'].str.len().mean():.0f} 字符")
    # 验证 query 都不含选项
    bad = sample["query"].str.contains("Answer Choices", na=False).sum()
    print(f"query 中残留 'Answer Choices' 的题数: {bad}（应为 0）")


if __name__ == "__main__":
    main()
