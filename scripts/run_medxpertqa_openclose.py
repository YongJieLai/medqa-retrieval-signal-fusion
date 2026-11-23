"""MedXpertQA 闭卷 vs 开卷：检索增益（should_retrieve）测量。

MedXpertQA 是 10 选 1（A-J），无 rationale 字段。

开卷证据设计（避免同源循环泄漏）：
  用「另一个模型」对正确答案生成 rationale（换源），喂给「被评测模型」。
  即：qwen-flash 生成解释 → 喂 qwen-plus 作答；qwen-plus 生成解释 → 喂 qwen-flash 作答。
  这样"生成解释的模型" ≠ "答题的模型"，缓解同源泄漏。

测量：
  correct_closed / correct_open / delta / should_retrieve（delta==+1）

输出：results/medxpertqa_openclose.jsonl
"""
from __future__ import annotations

import json
import os
import random
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


client = None


def call(model, prompt, max_tokens=10):
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return ""


def answer(model, question, options, rationale=None):
    opts_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    if rationale:
        p = (f"你是医学多选题助手。请结合下面给出的医学知识，选出唯一正确答案。\n"
             f"严格只输出一个字母（A/B/C/D/E/F/G/H/I/J），不要输出任何解释。\n\n"
             f"题干：{question}\n\n选项：\n{opts_lines}\n\n相关知识：\n{rationale}\n\n答案：")
    else:
        p = (f"你是医学多选题助手。请只根据题干和选项选出唯一正确答案。\n"
             f"严格只输出一个字母（A/B/C/D/E/F/G/H/I/J），不要输出任何解释。\n\n"
             f"题干：{question}\n\n选项：\n{opts_lines}\n\n答案：")
    t = call(model, p, max_tokens=10).upper()
    for ch in t:
        if ch in "ABCDEFGHIJ":
            return ch
    return "?"


def gen_rationale(model, question, options, answer_label):
    """用 model 对正确答案生成解释（换源证据）。"""
    opts_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    ans_text = options.get(answer_label, "")
    p = (f"下面是一道医学多选题。请用 2-3 句话解释：为什么选项 {answer_label}"
         f"（{ans_text}）是正确答案，背后的医学机制、诊断依据或治疗原理是什么。\n"
         f"只输出解释，不要复述题干。\n\n"
         f"题干：{question}\n\n选项：\n{opts_lines}\n\n解释：")
    return call(model, p, max_tokens=200)


def main():
    global client
    load_env()
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    rows = [json.loads(l) for l in open("data/medxpertqa_text.jsonl", encoding="utf-8")]

    # 抽样 40 题（Reasoning 20 + Understanding 20）
    rng = random.Random(7)
    reasoning = [r for r in rows if r["question_type"] == "Reasoning"]
    understanding = [r for r in rows if r["question_type"] == "Understanding"]
    sample = rng.sample(reasoning, 20) + rng.sample(understanding, 20)

    # 换源配对：flash 答题 ← plus 生成解释；plus 答题 ← flash 生成解释
    pairs = [
        ("qwen-plus", "qwen-flash-2025-07-28"),
        ("qwen-flash-2025-07-28", "qwen-plus"),
    ]

    out = []
    t0 = time.time()
    for answer_model, rationale_model in pairs:
        print(f"\n=== 答题模型 {answer_model}，解释来源 {rationale_model} ===", flush=True)
        stats = {"closed_ok": 0, "open_ok": 0, "retrieve": 0, "n": 0}
        for i, r in enumerate(sample):
            ans = r["label"]
            closed = answer(answer_model, r["question"], r["options"])
            cc = int(closed == ans)

            # 换源生成解释（仅对答错的题生成，省调用；答对的题不需要检索）
            if cc == 0:
                rationale = gen_rationale(rationale_model, r["question"], r["options"], ans)
                open_a = answer(answer_model, r["question"], r["options"], rationale)
                co = int(open_a == ans)
                delta = max(0, co - cc)
            else:
                rationale = ""
                co = cc
                delta = 0

            stats["closed_ok"] += cc
            stats["open_ok"] += co
            stats["retrieve"] += (1 if delta == 1 else 0)
            stats["n"] += 1

            out.append({
                "answer_model": answer_model, "rationale_model": rationale_model,
                "id": r["id"], "question_type": r["question_type"],
                "medical_task": r["medical_task"],
                "closed_pred": closed, "answer": ans,
                "correct_closed": cc, "correct_open": co,
                "delta": delta, "should_retrieve": delta,
                "rationale_len": len(rationale),
            })

            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(sample)}] 闭卷对={stats['closed_ok']} 开卷对={stats['open_ok']} 救回={stats['retrieve']} 耗时{round(el,0)}s", flush=True)

        print(f"  汇总: 闭卷 {stats['closed_ok']}/{stats['n']} = {stats['closed_ok']/stats['n']:.0%} | "
              f"开卷 {stats['open_ok']}/{stats['n']} | should_retrieve={stats['retrieve']}", flush=True)

    with open("results/medxpertqa_openclose.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已保存 {len(out)} 条到 results/medxpertqa_openclose.jsonl")


if __name__ == "__main__":
    main()
