"""BM25 检索器：对 MedRAG/pubmed 语料建 BM25 索引，检索相关段落。

内存友好（纯词频统计，无深度学习模型），作为 MedCPT 语义检索的对比/替代。

索引：前 N 个 chunk 的 title+content，BM25Okapi。
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time

sys.path.insert(0, ".")

from rank_bm25 import BM25Okapi

CHUNK_DIR = "kg/medrag_pubmed"
OUT_INDEX = f"{CHUNK_DIR}/bm25_index.pkl"
N_CHUNKS = 15  # 前 15 个 chunk ≈ 22 万篇


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def main():
    files = sorted(glob.glob(f"{CHUNK_DIR}/pubmed23n*.jsonl"))[:N_CHUNKS]
    print(f"建 BM25 索引：{len(files)} 个 chunk", flush=True)

    corpus = []
    meta = []
    t0 = time.time()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                text = f"{d.get('title', '')}. {d.get('content', '')}"
                corpus.append(tokenize(text))
                meta.append({"id": d.get("id"), "title": d.get("title", ""), "content": d.get("content", "")})

    print(f"语料 {len(corpus)} 篇，tokenize 完成，建 BM25...", flush=True)
    bm25 = BM25Okapi(corpus)

    # 保存（meta 单独存，避免 pickle 过大）
    with open(OUT_INDEX, "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{CHUNK_DIR}/bm25_meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    el = time.time() - t0
    print(f"完成：{len(corpus)} 篇，索引 {OUT_INDEX}，耗时 {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
