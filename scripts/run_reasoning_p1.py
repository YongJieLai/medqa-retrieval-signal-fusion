"""P1：推理增强方法评测 —— M0 单次 greedy + M1 CoT + M2 Self-Consistency。

按《推理增强方法评测计划书》：
- M0 单次 greedy：温度 0，一次解码直接选（旧基线）
- M1 CoT：温度 0，显式"逐步推理再作答"
- M2 Self-Consistency：温度 0.7，采样 N=10 次 CoT，多数投票（新公平基线）

输出：results/reasoning_p1.jsonl，每题一行，含 M0/M1 预测、M2 投票分布与最终预测。
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from medxpertqa_eval_framework import (
    call_llm, parse_answer, build_options_str, FORMAT_INSTRUCTION,
    get_client, DATA_200,
)

OUT = "results/reasoning_p1.jsonl"
N_SAMPLES = 10  # M2 采样次数


def build_m0_prompt(question, options):
    opts = build_options_str(options)
    return (
        "You are an expert physician. Answer the following multiple-choice medical question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )


def build_cot_prompt(question, options):
    opts = build_options_str(options)
    return (
        "You are an expert physician. Solve the following multiple-choice medical question "
        "step by step: first list the key clinical findings and your differential reasoning, "
        "then select the best answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )


def majority_vote(answers: list[str]) -> tuple[str, dict]:
    """多数投票，平票取最早达到最高票的选项。返回 (最终答案, 投票分布)。"""
    cnt = Counter(answers)
    # 按投票顺序找到最高票（平票取最早出现）
    order = list(dict.fromkeys(answers))
    best = order[0]
    for a in order:
        if cnt[a] > cnt[best]:
            best = a
    return best, dict(cnt)


def main():
    client, model = get_client()
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]

    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        qid = r["id"]
        question, options, label = r["question"], r["options"], r["label"]

        rec = {
            "id": qid,
            "question_type": r["question_type"],
            "medical_task": r["medical_task"],
            "body_system": r["body_system"],
            "label": label,
        }

        # M0 单次 greedy
        text_m0, tok_m0 = call_llm(client, model, build_m0_prompt(question, options), temperature=0.0)
        rec["M0_pred"] = parse_answer(text_m0)
        rec["M0_tokens"] = tok_m0["prompt"] + tok_m0["completion"]

        # M1 CoT
        text_m1, tok_m1 = call_llm(client, model, build_cot_prompt(question, options), temperature=0.0)
        rec["M1_pred"] = parse_answer(text_m1)
        rec["M1_tokens"] = tok_m1["prompt"] + tok_m1["completion"]

        # M2 Self-Consistency：采样 N 次
        votes = []
        vote_tokens = 0
        for _ in range(N_SAMPLES):
            text, tok = call_llm(client, model, build_cot_prompt(question, options),
                                 temperature=0.7, max_tokens=512)
            votes.append(parse_answer(text))
            vote_tokens += tok["prompt"] + tok["completion"]
        final_m2, dist = majority_vote(votes)
        rec["M2_pred"] = final_m2
        rec["M2_vote_dist"] = dist
        rec["M2_top_vote_ratio"] = max(dist.values()) / N_SAMPLES
        rec["M2_tokens"] = vote_tokens

        out.append(rec)

        if (idx + 1) % 10 == 0:
            el = time.time() - t0
            acc_m0 = sum(1 for x in out if x["M0_pred"] == x["label"]) / len(out)
            acc_m2 = sum(1 for x in out if x["M2_pred"] == x["label"]) / len(out)
            print(f"  [{idx+1}/{len(rows)}] 耗时 {round(el,0)}s  M0={acc_m0:.3f} M2={acc_m2:.3f}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # 汇总
    n = len(out)
    for mname in ["M0", "M1", "M2"]:
        acc = sum(1 for x in out if x[f"{mname}_pred"] == x["label"]) / n
        print(f"\n{mname}: {acc:.4f}", flush=True)
    print(f"\n已保存 {n} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
