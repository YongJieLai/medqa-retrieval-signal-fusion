"""P2：M2 推理查询扩展 + M5 证据过滤 + 查询重构 评测。

按计划书 §4.1：
- M2 推理查询扩展（先推理再检索）：
  1) LLM 生成"最可能诊断 + 关键推理步骤"（不答题）
  2) 用生成的诊断假设文本作为查询，BM25 检索 top-k
  3) 证据拼回原题，给出最终答案

- M5 证据过滤 + 查询重构：
  1) LLM 生成 rationale（候选诊断 + 推理）
  2) 检索 top-k 后，用 LLM 过滤无关/重复证据
  3) 用精炼证据回答

统一：qwen-plus，温度 0。防泄漏：检索 query 不含答案选项。

输出：results/p2_m2_m5.jsonl
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from medxpertqa_eval_framework import (
    BM25Retriever, call_llm, parse_answer, build_options_str,
    FORMAT_INSTRUCTION, get_client, save_results, DATA_200,
)

OUT = "results/p2_m2_m5.jsonl"
K = 3  # top-k 检索


# ---- M2 推理查询扩展 ----
def m2_generate_query(client, model, question, options):
    """第一步：生成诊断假设 + 推理步骤（不答题）。"""
    opts = build_options_str(options)
    prompt = (
        "Given the following clinical case, list (1) the most likely diagnosis, and (2) the key "
        "reasoning steps or differential diagnoses that would help reach the answer. "
        "Do NOT select an answer choice yet.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n"
    )
    text, tok = call_llm(client, model, prompt, max_tokens=400)
    return text, tok


def m2_answer(client, model, question, options, passages):
    """第三步：证据拼回原题答题。"""
    opts = build_options_str(options)
    prompt = (
        "You are an expert physician. Answer the following multiple-choice medical question "
        "using the provided evidence.\n\n"
        f"Relevant evidence:\n{passages}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )
    text, tok = call_llm(client, model, prompt)
    return text, tok


# ---- M5 证据过滤 + 查询重构 ----
def m5_generate_rationale(client, model, question, options):
    """第一步：生成 rationale（候选诊断 + 推理），作为重构查询。"""
    opts = build_options_str(options)
    prompt = (
        "Given the clinical case below, generate a brief rationale: the candidate diagnoses and "
        "the key clinical reasoning. Do NOT select an answer choice.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n"
    )
    text, tok = call_llm(client, model, prompt, max_tokens=400)
    return text, tok


def m5_filter_evidence(client, model, question, passages_list):
    """第二步：对每条检索证据判断相关性，过滤无关证据。"""
    kept = []
    for p in passages_list:
        prompt = (
            "Given the clinical question below, is this retrieved passage useful for answering it? "
            "Answer only YES or NO.\n\n"
            f"Question: {question}\n"
            f"Passage: {p['title']}. {p['content']}\n"
        )
        text, tok = call_llm(client, model, prompt, max_tokens=5)
        if text.strip().upper().startswith("YES"):
            kept.append(p)
    return kept


def m5_answer(client, model, question, options, passages):
    """第三步：用精炼证据答题。"""
    return m2_answer(client, model, question, options, passages)


def main():
    client, model = get_client()
    retriever = BM25Retriever()
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]

    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        qid = r["id"]
        question = r["question"]
        query = r["query"]
        options = r["options"]
        label = r["label"]

        rec = {
            "id": qid,
            "question_type": r["question_type"],
            "medical_task": r["medical_task"],
            "body_system": r["body_system"],
            "label": label,
        }

        # === M2 推理查询扩展 ===
        t = time.time()
        diag_text, tok_diag = m2_generate_query(client, model, question, options)
        rec["M2_query"] = diag_text[:300]
        rec["M2_query_tokens"] = tok_diag["prompt"] + tok_diag["completion"]

        top3_m2 = retriever.retrieve(diag_text, k=K)
        rec["M2_top3_titles"] = [p["title"][:80] for p in top3_m2]
        passages_m2 = retriever.format_passages(top3_m2)
        ans_m2, tok_m2 = m2_answer(client, model, question, options, passages_m2)
        rec["M2_pred"] = parse_answer(ans_m2)
        rec["M2_tokens"] = tok_diag["prompt"] + tok_diag["completion"] + tok_m2["prompt"] + tok_m2["completion"]
        rec["M2_latency_ms"] = int((time.time() - t) * 1000)

        # === M5 证据过滤 + 查询重构 ===
        t = time.time()
        rationale, tok_rat = m5_generate_rationale(client, model, question, options)
        rec["M5_rationale"] = rationale[:300]

        top5_m5 = retriever.retrieve(rationale, k=5)
        kept = m5_filter_evidence(client, model, question, top5_m5)
        rec["M5_kept_count"] = len(kept)
        rec["M5_retrieved_count"] = len(top5_m5)
        passages_m5 = retriever.format_passages(kept) if kept else "(no relevant evidence retained)"
        ans_m5, tok_m5 = m5_answer(client, model, question, options, passages_m5)
        rec["M5_pred"] = parse_answer(ans_m5)
        rec["M5_tokens"] = tok_rat["prompt"] + tok_rat["completion"] + tok_m5["prompt"] + tok_m5["completion"]
        rec["M5_latency_ms"] = int((time.time() - t) * 1000)

        out.append(rec)

        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{idx+1}/{len(rows)}] 耗时 {round(el,0)}s", flush=True)

    save_results(out, OUT)

    # 汇总（需结合 P1 的 M0 baseline）
    acc_m2 = sum(1 for x in out if x["M2_pred"] == x["label"]) / len(out)
    acc_m5 = sum(1 for x in out if x["M5_pred"] == x["label"]) / len(out)
    print("\n=== P2 结果 ===", flush=True)
    print(f"  M2（推理查询扩展）: {acc_m2:.4f}", flush=True)
    print(f"  M5（证据过滤+重构）: {acc_m5:.4f}", flush=True)
    print(f"  M5 平均保留证据数: {sum(x['M5_kept_count'] for x in out)/len(out):.2f}", flush=True)
    print(f"\n已保存 {len(out)} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
