"""用 MedQA 真实数据做闭卷筛选，找出 qwen-plus 答错的题（难样本）。

目标：坐实"模型侧失明"——找出 qwen-plus 闭卷答错的题，
这些题才有检索边际增益（should_retrieve=1 的候选）。

用法：
  python scripts/medqa_closed_screen.py --n 100 --out results/medqa_closed.jsonl
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
    parser.add_argument("--n", type=int, default=100, help="闭卷作答的题数")
    parser.add_argument("--out", type=str, default="results/medqa_closed.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()
    client = LLMClient()  # qwen-plus

    # 读本地缓存
    rows = []
    with open("data/medqa_test.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # 采样：step1 与 step2&3 各半，保证难度分层
    import random
    rng = random.Random(args.seed)
    step1 = [r for r in rows if r["meta_info"] == "step1"]
    step23 = [r for r in rows if r["meta_info"] == "step2&3"]
    n1 = min(len(step1), args.n // 2)
    n23 = min(len(step23), args.n - n1)
    sample = rng.sample(step1, n1) + rng.sample(step23, n23)
    rng.shuffle(sample)

    os.makedirs("results", exist_ok=True)
    results = []
    correct = 0
    for i, r in enumerate(sample):
        opts = [r["options"][k] for k in "ABCD"]
        pred = client.answer_closed(r["question"], opts)
        ok = pred == r["answer_idx"]
        correct += int(ok)
        results.append({
            "id": r["id"],
            "meta_info": r["meta_info"],
            "pred": pred,
            "answer_idx": r["answer_idx"],
            "correct": int(ok),
            "metamap_phrases": r["metamap_phrases"],
        })
        if (i + 1) % 20 == 0:
            print(f"进度 {i+1}/{len(sample)}, 当前正确率 {correct}/{i+1} = {correct/(i+1):.1%}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    wrong = [r for r in results if not r["correct"]]
    print(f"\n=== 结果 ===")
    print(f"总题数: {len(results)}, 闭卷正确率: {correct}/{len(results)} = {correct/len(results):.1%}")
    print(f"答错题数（难样本候选）: {len(wrong)}")
    print(f"step1 答错: {sum(1 for r in wrong if r['meta_info']=='step1')}")
    print(f"step2&3 答错: {sum(1 for r in wrong if r['meta_info']=='step2&3')}")


if __name__ == "__main__":
    main()
