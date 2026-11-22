"""构造混合长尾批次：难样本（should_retrieve=1）+ 易样本（should_retrieve=0）。

难样本：来自 MedQA hard 子集 ground truth（qwen-plus 闭卷答错、开卷答对 → should_retrieve=1）。
易样本：来自全 MedQA test 集，qwen-plus 闭卷答对（→ should_retrieve=0，无需开卷验证，
        因为已答对，Δ=max(0,1-1)=0）。

混合目标：约 50 题，难:易 均衡，类型1(step1)/类型2(step2&3) 均衡。

输出：results/mixed_batch_gt.jsonl（含完整 ground truth + 类型标签）
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, ".")

from src.execute.llm_client import LLMClient


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    rng = random.Random(42)

    # 1. 难样本：hard 子集 should_retrieve=1 的题
    hard = [json.loads(l) for l in open("results/medqa_hard_gt.jsonl", encoding="utf-8")]
    hard_pos = [r for r in hard if r["should_retrieve"] == 1]
    print(f"难样本（should_retrieve=1）可用: {len(hard_pos)} 题")

    # 2. 易样本：从 medqa_test 找 qwen-plus 闭卷答对的题
    #    先用已有的 300 题闭卷筛选结果（有 correct 标签），避免重复调用
    closed = [json.loads(l) for l in open("results/medqa_closed_300.jsonl", encoding="utf-8")]
    easy_candidates = [r for r in closed if r["correct"] == 1]
    print(f"易样本候选（闭卷答对）: {len(easy_candidates)} 题")

    # 易样本的 should_retrieve=0 是确定的（已答对），但需要补全 question/options/answer 字段
    # closed 筛选结果只有 id/pred/answer_idx/correct/metamap，缺 question/options
    # 从 medqa_test.jsonl 回查完整字段
    test_map = {}
    for l in open("data/medqa_test.jsonl", encoding="utf-8"):
        r = json.loads(l)
        test_map[r["id"]] = r

    # 3. 采样：难 25 题 + 易 25 题 = 50 题混合批次
    n_hard = 25
    n_easy = 25
    hard_sample = rng.sample(hard_pos, n_hard)
    easy_sample = rng.sample(easy_candidates, n_easy)

    # 组装
    batch = []
    for r in hard_sample:
        batch.append(r)  # hard 的 gt 已含完整字段

    for r in easy_sample:
        full = test_map[r["id"]]
        batch.append({
            "id": full["id"],
            "meta_info": full["meta_info"],
            "question": full["question"],
            "options": full["options"],
            "answer_idx": full["answer_idx"],
            "closed_pred": r["pred"],
            "open_pred": r["pred"],  # 已答对，开卷也答对（Δ=0）
            "correct_closed": 1,
            "correct_open": 1,
            "delta": 0,
            "should_retrieve": 0,
            "rationale": "",
            "metamap_phrases": full["metamap_phrases"],
        })

    rng.shuffle(batch)
    os.makedirs("results", exist_ok=True)
    with open("results/mixed_batch_gt.jsonl", "w", encoding="utf-8") as f:
        for r in batch:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\n=== 混合批次构造完成 ===")
    print(f"总题数: {len(batch)}")
    print(f"should_retrieve=1（难/该检索）: {sum(1 for r in batch if r['should_retrieve']==1)}")
    print(f"should_retrieve=0（易/不该检索）: {sum(1 for r in batch if r['should_retrieve']==0)}")
    print(f"类型分布: {Counter(r['meta_info'] for r in batch)}")


if __name__ == "__main__":
    main()
