"""语义检索导出脚本（在云服务器上运行）。

对 query 文件里的三种 query（raw_query / m2_query / m5_query）分别用 MedCPT-Query-Encoder
编码 + FAISS IndexFlatIP 检索 top-k，导出结果 jsonl。

输出：semantic_retrieval.jsonl，每行 = {id, raw_query: [...], m2_query: [...], m5_query: [...]}
每条结果 = {idx, score, title, content}
"""
from __future__ import annotations

import json
import sys

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

INDEX_PATH = "kg/medrag_pubmed/index.faiss"
META_PATH = "kg/medrag_pubmed/corpus_meta.jsonl"
QUERY_ENCODER = "kg/medcpt/MedCPT-Query-Encoder"
INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/p2_queries_for_retrieval.jsonl"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "results/semantic_retrieval.jsonl"
K = 5


def main():
    # 加载索引和语料
    index = faiss.read_index(INDEX_PATH)
    print(f"[index] {index.ntotal} 篇", flush=True)

    # 加载 meta（与索引行对齐）
    meta = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]

    # 加载 Query Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(QUERY_ENCODER)
    model = AutoModel.from_pretrained(QUERY_ENCODER)
    if device == "cuda":
        model = model.half().to(device)
    model.eval()
    print(f"[encoder] device={device}", flush=True)

    def encode_query(query: str) -> np.ndarray:
        enc = tok(query, return_tensors="pt", max_length=512, truncation=True, padding=True)
        if device == "cuda":
            enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def retrieve(query: str, k: int = K) -> list[dict]:
        q = encode_query(query)
        scores, idxs = index.search(q, k)
        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0 or i >= len(meta):
                continue
            m = meta[i]
            results.append({
                "idx": int(i),
                "score": float(s),
                "title": m.get("title", ""),
                "content": m.get("content", ""),
            })
        return results

    rows = [json.loads(l) for l in open(INPUT, encoding="utf-8")]
    print(f"[input] {len(rows)} 条 query，top-k={K}", flush=True)

    out_f = open(OUTPUT, "w", encoding="utf-8")
    for i, r in enumerate(rows):
        rec = {"id": r["id"]}
        for key in ["raw_query", "m2_query", "m5_query"]:
            q = r.get(key, "") or ""
            rec[key] = retrieve(q, K)
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    out_f.close()
    print(f"完成，输出 {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
