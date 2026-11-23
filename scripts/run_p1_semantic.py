"""P1 语义检索版：M1 标准 RAG 用 MedCPT 语义检索（173万篇 FAISS）重跑。

复用 P1 已生成的 M0 基线（闭卷），只重跑 M1 k∈{1,3,5} 用语义检索结果。
防泄漏：语义检索用 raw_query（去选项题干）。

输出：results/p1_semantic.jsonl
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

OUT = "results/p1_semantic.jsonl"
SEMANTIC = "results/semantic_retrieval.jsonl"
K_VALUES = [1, 3, 5]


def load_semantic():
    return {json.loads(l)["id"]: json.loads(l) for l in open(SEMANTIC, encoding="utf-8")}


def format_passages(results: list[dict]) -> str:
    parts = []
    for j, r in enumerate(results, 1):
        parts.append(f"[{j}] {r['title']}. {r['content']}")
    return "\n".join(parts)


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
    sem = load_semantic()
    p1 = {json.loads(l)["id"]: json.loads(l) for l in open("results/p1_m0_m1.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]

    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        qid = r["id"]
        question, options, label = r["question"], r["options"], r["label"]
        sem_rec = sem[qid]
        top5 = sem_rec["raw_query"][:5]

        rec = {
            "id": qid,
            "question_type": r["question_type"],
            "medical_task": r["medical_task"],
            "body_system": r["body_system"],
            "label": label,
            "M0_pred": p1[qid]["M0_pred"],  # 复用 M0 基线
            "M1_top5_titles": [p["title"][:80] for p in top5],
            "M1_top5_scores": [round(p["score"], 3) for p in top5],
        }

        for k in K_VALUES:
            passages = format_passages(top5[:k])
            t = time.time()
            text_k, tok_k = call_llm(client, model, build_m1_prompt(question, options, passages))
            rec[f"M1_k{k}_pred"] = parse_answer(text_k)
            rec[f"M1_k{k}_tokens"] = tok_k["prompt"] + tok_k["completion"]
            rec[f"M1_k{k}_latency_ms"] = int((time.time() - t) * 1000)

        out.append(rec)

        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            acc_m1 = sum(1 for x in out if x["M1_k3_pred"] == x["label"]) / len(out)
            print(f"  [{idx+1}/{len(rows)}] 耗时 {round(el,0)}s  M1_k3 acc={acc_m1:.3f}", flush=True)

    save_results(out, OUT)

    acc = {}
    for mname in ["M0"] + [f"M1_k{k}" for k in K_VALUES]:
        pred_key = f"{mname}_pred"
        correct = sum(1 for x in out if x[pred_key] == x["label"])
        acc[mname] = round(correct / len(out), 4)

    print("\n=== P1 语义检索版结果 ===", flush=True)
    for mname, a in acc.items():
        delta = a - acc["M0"]
        print(f"  {mname}: {a:.4f} (Δ={delta:+.4f})", flush=True)
    print(f"\n已保存 {len(out)} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
