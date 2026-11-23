"""第一步：LLM 抽取 MedXpertQA 200 题的医学实体。

输出：results/medxpertqa_entities.jsonl
每行 = {id, question_type, medical_task, body_system, entities: [str]}

实体用于后续计算知识侧 coverage（查 PrimeKG 稀有度）。
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from medxpertqa_eval_framework import get_client, call_llm, DATA_200

OUT = "results/medxpertqa_entities.jsonl"


def extract_entities(client, model, question, max_n=8):
    prompt = (
        "Extract the most important medical entities (disease names, symptoms, anatomical sites, "
        "drugs, tests, pathogens, lab findings, etc.) from the medical question below.\n"
        "Output ONLY the entity names in English, separated by commas, at most 8 entities. "
        "No explanation, no article text.\n\n"
        f"Question: {question}\n\nMedical entities:"
    )
    text, _ = call_llm(client, model, prompt, max_tokens=120)
    if not text:
        return []
    entities = [e.strip() for e in text.replace("\n", ",").split(",")]
    entities = [e for e in entities if len(e) >= 2]
    return entities[:max_n]


def main():
    client, model = get_client()
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]

    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        entities = extract_entities(client, model, r["question"])
        out.append({
            "id": r["id"],
            "question_type": r["question_type"],
            "medical_task": r["medical_task"],
            "body_system": r["body_system"],
            "entities": entities,
        })
        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{idx+1}/{len(rows)}] 耗时 {round(el,0)}s", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # 统计
    nlens = [len(o["entities"]) for o in out]
    print(f"\n完成，平均实体数 {sum(nlens)/len(nlens):.2f}，已保存 {OUT}", flush=True)
    print("样例：")
    for o in out[:3]:
        print(f"  {o['id']}: {o['entities']}")


if __name__ == "__main__":
    main()
