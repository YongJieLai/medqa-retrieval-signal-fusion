"""P2 语义检索版：用 MedCPT 语义检索（173万篇 FAISS）替换 BM25，重跑 M2/M5 答题。

关键：复用 P2 已生成的 M2_query（诊断假设）和 M5_rationale（候选诊断理由），
只替换检索数据源为语义检索结果，重新执行答题步骤（m2_answer / m5_answer）。

不再重新生成查询，大幅节省成本。防泄漏：语义检索 query 用的是已去选项的
M2_query/M5_rationale，不含答案选项。

输出：results/p2_semantic.jsonl
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from medxpertqa_eval_framework import (
    call_llm, parse_answer, build_options_str, FORMAT_INSTRUCTION,
    get_client, save_results, DATA_200,
)

OUT = "results/p2_semantic.jsonl"
SEMANTIC = "results/semantic_retrieval.jsonl"
K = 3


def load_semantic():
    return {json.loads(l)["id"]: json.loads(l) for l in open(SEMANTIC, encoding="utf-8")}


def format_passages(results: list[dict]) -> str:
    parts = []
    for j, r in enumerate(results, 1):
        parts.append(f"[{j}] {r['title']}. {r['content']}")
    return "\n".join(parts)


def answer_with_evidence(client, model, question, options, passages):
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


def main():
    client, model = get_client()
    sem = load_semantic()
    p2 = [json.loads(l) for l in open("results/p2_m2_m5.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]
    data = {r["id"]: r for r in rows}

    out = []
    t0 = time.time()
    for idx, r in enumerate(p2):
        qid = r["id"]
        d = data[qid]
        question, options, label = d["question"], d["options"], d["label"]
        sem_rec = sem[qid]

        rec = {
            "id": qid,
            "question_type": d["question_type"],
            "medical_task": d["medical_task"],
            "body_system": d["body_system"],
            "label": label,
            "M2_query": r["M2_query"],          # 复用已生成的诊断假设
            "M5_rationale": r["M5_rationale"],  # 复用已生成的 rationale
        }

        # === M2：用语义检索 m2_query 的结果答题 ===
        top_m2 = sem_rec["m2_query"][:K]
        rec["M2_top3_titles"] = [p["title"][:80] for p in top_m2]
        rec["M2_top3_scores"] = [round(p["score"], 3) for p in top_m2]
        ans_m2, tok_m2 = answer_with_evidence(client, model, question, options, format_passages(top_m2))
        rec["M2_pred"] = parse_answer(ans_m2)
        rec["M2_tokens"] = tok_m2["prompt"] + tok_m2["completion"]
        rec["M2_latency_ms"] = int(tok_m2["latency_ms"]) if "latency_ms" in tok_m2 else 0

        # === M5：用语义检索 m5_query 的结果，LLM 过滤后答题 ===
        top_m5 = sem_rec["m5_query"][:5]
        rec["M5_retrieved_count"] = len(top_m5)
        # LLM 过滤证据
        kept = []
        for p in top_m5:
            prompt = (
                "Given the clinical question below, is this retrieved passage useful for answering it? "
                "Answer only YES or NO.\n\n"
                f"Question: {question}\n"
                f"Passage: {p['title']}. {p['content']}\n"
            )
            text, _ = call_llm(client, model, prompt, max_tokens=5)
            if text.strip().upper().startswith("YES"):
                kept.append(p)
        rec["M5_kept_count"] = len(kept)
        passages_m5 = format_passages(kept) if kept else "(no relevant evidence retained)"
        ans_m5, tok_m5 = answer_with_evidence(client, model, question, options, passages_m5)
        rec["M5_pred"] = parse_answer(ans_m5)
        rec["M5_tokens"] = tok_m5["prompt"] + tok_m5["completion"]

        out.append(rec)

        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{idx+1}/{len(p2)}] 耗时 {round(el,0)}s", flush=True)

    save_results(out, OUT)

    acc_m2 = sum(1 for x in out if x["M2_pred"] == x["label"]) / len(out)
    acc_m5 = sum(1 for x in out if x["M5_pred"] == x["label"]) / len(out)
    print("\n=== P2 语义检索版结果 ===", flush=True)
    print(f"  M2（推理查询扩展 + 语义检索）: {acc_m2:.4f}", flush=True)
    print(f"  M5（证据过滤 + 语义检索）: {acc_m5:.4f}", flush=True)
    print(f"  M5 平均保留证据数: {sum(x['M5_kept_count'] for x in out)/len(out):.2f}", flush=True)
    print(f"\n已保存 {len(out)} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
