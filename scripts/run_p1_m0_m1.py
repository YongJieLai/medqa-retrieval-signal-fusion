"""P1：M0 Baseline（无检索）+ M1 标准 RAG（BM25 top-k）评测。

按计划书 §4.1 实现：
- M0：直接 prompt(q) → 答案（附 A 的 Baseline prompt）
- M1：检索(q) → top-k 证据拼接 → prompt(q, 证据) → 答案（k ∈ {1,3,5}）

统一：qwen-plus，温度 0，`Answer: <LETTER>` 格式抽取。
防泄漏：检索用去选项的 query 字段。

输出：results/p1_m0_m1.jsonl（每题一行，含各配置预测 + token + 延迟）
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from medxpertqa_eval_framework import (
    BM25Retriever, call_llm, parse_answer, build_options_str,
    FORMAT_INSTRUCTION, get_client, save_results, DATA_200,
)

OUT = "results/p1_m0_m1.jsonl"
K_VALUES = [1, 3, 5]


def build_m0_prompt(question, options):
    opts = build_options_str(options)
    return (
        "You are an expert physician. Answer the following multiple-choice medical question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )


def build_m1_prompt(question, options, passages):
    opts = build_options_str(options)
    return (
        "You are an expert physician. Answer the following multiple-choice medical question "
        "using the provided evidence.\n\n"
        f"Relevant evidence:\n{passages}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )


def main():
    client, model = get_client()
    retriever = BM25Retriever()
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]

    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        qid = r["id"]
        question = r["question"]
        query = r["query"]  # 去选项，防泄漏
        options = r["options"]
        label = r["label"]

        rec = {
            "id": qid,
            "question_type": r["question_type"],
            "medical_task": r["medical_task"],
            "body_system": r["body_system"],
            "label": label,
        }

        # M0 Baseline
        t = time.time()
        text_m0, tok_m0 = call_llm(client, model, build_m0_prompt(question, options))
        rec["M0_pred"] = parse_answer(text_m0)
        rec["M0_tokens"] = tok_m0["prompt"] + tok_m0["completion"]
        rec["M0_latency_ms"] = int((time.time() - t) * 1000)

        # M1 标准 RAG（检索一次 top-5，截断到 k）
        t = time.time()
        top5 = retriever.retrieve(query, k=5)
        rec["M1_retrieval_ms"] = int((time.time() - t) * 1000)
        rec["M1_top5_titles"] = [p["title"][:80] for p in top5]

        for k in K_VALUES:
            passages = retriever.format_passages(top5[:k])
            t = time.time()
            text_k, tok_k = call_llm(client, model, build_m1_prompt(question, options, passages))
            rec[f"M1_k{k}_pred"] = parse_answer(text_k)
            rec[f"M1_k{k}_tokens"] = tok_k["prompt"] + tok_k["completion"]
            rec[f"M1_k{k}_latency_ms"] = int((time.time() - t) * 1000)

        out.append(rec)

        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            # 即时准确率
            acc_m0 = sum(1 for x in out if x["M0_pred"] == x["label"]) / len(out)
            print(f"  [{idx+1}/{len(rows)}] 耗时 {round(el,0)}s  M0 acc={acc_m0:.3f}", flush=True)

    save_results(out, OUT)

    # 汇总
    acc = {}
    for mname in ["M0"] + [f"M1_k{k}" for k in K_VALUES]:
        pred_key = f"{mname}_pred"
        correct = sum(1 for x in out if x[pred_key] == x["label"])
        acc[mname] = round(correct / len(out), 4)

    print("\n=== P1 结果 ===", flush=True)
    for mname, a in acc.items():
        delta = a - acc["M0"]
        print(f"  {mname}: {a:.4f} (Δ={delta:+.4f})", flush=True)
    print(f"\n已保存 {len(out)} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
