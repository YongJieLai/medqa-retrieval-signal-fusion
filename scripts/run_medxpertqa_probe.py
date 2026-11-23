"""千问各模型在 MedXpertQA 上的闭卷正确率探测。

MedXpertQA：10 选 1（A-J），question_type = Reasoning / Understanding。

跑多个千问模型，对比：
  1. 整体正确率（10 选 1 随机 = 10%）
  2. Reasoning vs Understanding 两类题的正确率差异（知识型 vs 推理型）
  3. 是否出现"失明"（正确率远低于模型应有水平）

输出：results/medxpertqa_closed_probe.jsonl
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time

sys.path.insert(0, ".")

from openai import OpenAI


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


MODELS = ["qwen-plus", "qwen-flash-2025-07-28", "qwen3.5-27b"]


def answer(client, model, question, options):
    """10 选 1 作答，返回字母 A-J。"""
    opts_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    p = (f"你是医学多选题助手。请根据题干和选项选出唯一正确答案。\n"
         f"严格只输出一个字母（A/B/C/D/E/F/G/H/I/J），不要输出任何解释。\n\n"
         f"题干：{question}\n\n选项：\n{opts_lines}\n\n答案：")
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": p}],
                temperature=0.0, max_tokens=10)
            t = r.choices[0].message.content.strip().upper()
            # 提取第一个 A-J 字母
            for ch in t:
                if ch in "ABCDEFGHIJ":
                    return ch
            return "?"
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return "?"


def main():
    load_env()
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    rows = [json.loads(l) for l in open("data/medxpertqa_text.jsonl", encoding="utf-8")]

    # 抽样：Reasoning 和 Understanding 各 15 题（共 30 题，每个模型 30 次作答）
    rng = random.Random(42)
    reasoning = [r for r in rows if r["question_type"] == "Reasoning"]
    understanding = [r for r in rows if r["question_type"] == "Understanding"]
    sample = rng.sample(reasoning, 15) + rng.sample(understanding, 15)

    out = []
    with open("results/medxpertqa_closed_probe.jsonl", "w", encoding="utf-8") as f:
        for model in MODELS:
            print(f"\n=== 模型 {model} ===", flush=True)
            correct = {"all": 0, "Reasoning": 0, "Understanding": 0}
            total = {"all": 0, "Reasoning": 0, "Understanding": 0}
            t0 = time.time()
            for i, r in enumerate(sample):
                pred = answer(client, model, r["question"], r["options"])
                ans = r["label"]
                ok = int(pred == ans)
                qt = r["question_type"]
                correct["all"] += ok
                correct[qt] += ok
                total["all"] += 1
                total[qt] += 1
                rec = {"model": model, "id": r["id"], "question_type": qt,
                       "pred": pred, "answer": ans, "correct": ok}
                out.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{len(sample)}] 当前整体 {correct['all']}/{total['all']} 耗时 {round(time.time()-t0,0)}s", flush=True)
            print(f"  整体: {correct['all']}/{total['all']} = {correct['all']/total['all']:.0%}", flush=True)
            print(f"  Reasoning: {correct['Reasoning']}/{total['Reasoning']} = {correct['Reasoning']/total['Reasoning']:.0%}", flush=True)
            print(f"  Understanding: {correct['Understanding']}/{total['Understanding']} = {correct['Understanding']/total['Understanding']:.0%}", flush=True)

    print(f"\n已保存 {len(out)} 条到 results/medxpertqa_closed_probe.jsonl")


if __name__ == "__main__":
    main()
