"""P0 失效基线：MedCPT 检索（现场编码的 PubMed 索引）+ qwen-flash 8 配置推理。

检索器：MedCPT Query Encoder（kg/medcpt）+ 现场编码的 FAISS 索引（kg/medrag_pubmed/index.faiss）。

配置（按清单 §3）：
- M0a Direct / M0b CoT：闭卷基线
- M1a/b/c：Standard RAG，MedCPT 检索 top-k (k=4/8/16)
- M2a/b/c：Enhanced RAG（改写 / 过滤 / 改写+过滤）

防泄漏：检索 query 只用题干主体（去选项），全程不碰 label。

输出：results/results_p0.jsonl（每题一行 × 8 配置）
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DATA_200 = "data/medxpertqa_p0_200.jsonl"
OUT = "results/results_p0.jsonl"
MODEL = "qwen-flash-2025-07-28"  # 弱 backbone（替代已下线的 qwen3-8b）
K_VALUES = [4, 8, 16]

INDEX_PATH = "kg/medrag_pubmed/index.faiss"
META_PATH = "kg/medrag_pubmed/corpus_meta.jsonl"


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class MedCPTRetriever:
    def __init__(self):
        # FAISS 索引
        self.index = faiss.read_index(INDEX_PATH)
        # corpus 元数据（与索引行对齐）
        self.meta = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]
        # Query Encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("kg/medcpt/MedCPT-Query-Encoder")
        self.model = AutoModel.from_pretrained("kg/medcpt/MedCPT-Query-Encoder")
        if self.device == "cuda":
            self.model = self.model.half().to(self.device)
        self.model.eval()
        print(f"[retriever] 索引 {self.index.ntotal} 篇，语料 {len(self.meta)} 条，device={self.device}", flush=True)

    def encode_query(self, query: str) -> np.ndarray:
        enc = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding=True)
        if self.device == "cuda":
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def retrieve(self, query: str, k: int = 16) -> list[dict]:
        q = self.encode_query(query)
        scores, idxs = self.index.search(q, k)
        results = []
        for s, i in zip(scores[0], idxs[0]):
            m = self.meta[i]
            results.append({"idx": i, "score": float(s), "title": m["title"], "content": m["content"]})
        return results

    @staticmethod
    def format_passages(results: list[dict]) -> str:
        parts = []
        for j, r in enumerate(results, 1):
            parts.append(f"[{j}] {r['title']}. {r['content']}")
        return "\n".join(parts)


def call_llm(client, prompt, max_tokens=20):
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return ""


def parse_answer(text: str) -> str:
    for ch in text.upper():
        if ch in "ABCDEFGHIJ":
            return ch
    return "?"


def build_options_str(options: dict) -> str:
    return "\n".join(f"({k}) {v}" for k, v in options.items())


def answer_closed(client, question, options, cot=False):
    opts = build_options_str(options)
    if cot:
        prompt = (f"Question: {question}\nOptions: {opts}\n"
                  f"Let's think step by step, then give the final answer as a single letter (A-J).")
    else:
        prompt = (f"Question: {question}\nOptions: {opts}\n"
                  f"Answer with only the single letter (A-J) of the correct option.")
    return call_llm(client, prompt, max_tokens=300 if cot else 20)


def answer_rag(client, question, options, passages):
    opts = build_options_str(options)
    prompt = (f"Retrieved passages:\n{passages}\n\n"
              f"Question: {question}\nOptions: {opts}\n"
              f"Based on the retrieved passages, answer with only the single letter (A-J).")
    return call_llm(client, prompt, max_tokens=20)


def gen_reasoning_trace(client, question):
    prompt = (f"Given the medical question below, write a short reasoning trace about what key clinical "
              f"facts and differential diagnoses are relevant to solving it (do NOT give the answer letter):\n"
              f"Question: {question}\nReasoning trace:")
    return call_llm(client, prompt, max_tokens=300)


def main():
    load_env()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]
    retriever = MedCPTRetriever()

    out = []
    t0 = time.time()

    for idx, r in enumerate(rows):
        qid = r["id"]
        question = r["question"]
        query = r["query"]
        options = r["options"]
        label = r["label"]

        rec = {"id": qid, "question_type": r["question_type"],
               "medical_task": r["medical_task"], "body_system": r["body_system"],
               "label": label}

        # M0 闭卷
        for mname, cot in [("M0a", False), ("M0b", True)]:
            ans_text = answer_closed(client, question, options, cot=cot)
            rec[f"{mname}_pred"] = parse_answer(ans_text)

        # M1 Standard RAG（检索一次 top-16，截断到 k）
        top16 = retriever.retrieve(query, k=16)
        rec["top_passages"] = [{"title": p["title"][:120], "score": round(p["score"], 3)} for p in top16[:8]]
        for k in K_VALUES:
            passages = retriever.format_passages(top16[:k])
            ans_text = answer_rag(client, question, options, passages)
            rec[f"M1_k{k}_pred"] = parse_answer(ans_text)

        # M2 Enhanced RAG
        trace = gen_reasoning_trace(client, question)
        top16_reform = retriever.retrieve(trace, k=16)

        # M2a 改写（reasoning trace 检索）
        passages_reform = retriever.format_passages(top16_reform[:16])
        rec["M2a_pred"] = parse_answer(answer_rag(client, question, options, passages_reform))

        # M2b 过滤（top-16 按分数取 top-3）
        top3 = sorted(top16, key=lambda x: -x["score"])[:3]
        rec["M2b_pred"] = parse_answer(answer_rag(client, question, options, retriever.format_passages(top3)))

        # M2c 改写+过滤
        top3_reform = sorted(top16_reform, key=lambda x: -x["score"])[:3]
        rec["M2c_pred"] = parse_answer(answer_rag(client, question, options, retriever.format_passages(top3_reform)))

        out.append(rec)

        if (idx + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{idx+1}/{len(rows)}] 耗时 {round(el,0)}s", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n已保存 {len(out)} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
