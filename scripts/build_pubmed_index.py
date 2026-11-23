"""流式编码 MedRAG/pubmed 语料，建 FAISS 索引（内存友好）。

逐 chunk 读、逐 batch 编码、增量写入索引，不一次性加载全部语料。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, ".")

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

CHUNK_DIR = "kg/medrag_pubmed"
OUT_INDEX = f"{CHUNK_DIR}/index.faiss"
OUT_META = f"{CHUNK_DIR}/corpus_meta.jsonl"
BATCH = 256
MAX_LEN = 256
DIM = 768


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("kg/medcpt/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("kg/medcpt/MedCPT-Article-Encoder")
    if device == "cuda":
        model = model.half().to(device)
    model.eval()
    print(f"[encoder] device={device}  batch={BATCH}  max_len={MAX_LEN}", flush=True)
    return tok, model, device


def encode_batch(model, tok, texts, device):
    enc = tok(texts, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding=True)
    if device == "cuda":
        enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    return emb


def main():
    files = sorted(glob.glob(f"{CHUNK_DIR}/pubmed23n*.jsonl"))
    print(f"发现 {len(files)} 个 chunk", flush=True)

    tok, model, device = load_model()

    index = faiss.IndexFlatIP(DIM)
    meta_f = open(OUT_META, "w", encoding="utf-8")
    global_idx = 0
    t0 = time.time()

    for fi, f in enumerate(files):
        # 读单个 chunk 的所有行
        batch_texts = []
        batch_meta = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                title = d.get("title", "") or ""
                content = d.get("content", "") or ""
                batch_texts.append(f"{title}. {content}")
                batch_meta.append({"id": d.get("id"), "title": title, "content": content})

        # 编码该 chunk
        n_chunk = len(batch_texts)
        for start in range(0, n_chunk, BATCH):
            texts = batch_texts[start:start + BATCH]
            metas = batch_meta[start:start + BATCH]
            emb = encode_batch(model, tok, texts, device)
            index.add(emb)
            for m in metas:
                meta_f.write(json.dumps({"idx": global_idx, **m}, ensure_ascii=False) + "\n")
                global_idx += 1

        el = time.time() - t0
        rate = global_idx / el
        print(f"  [{fi+1}/{len(files)}] chunk={f.split('/')[-1]} 累计 {global_idx} 篇 速率 {rate:.0f}篇/s 已用时 {el/60:.1f}min", flush=True)

    meta_f.close()
    faiss.write_index(index, OUT_INDEX)
    el = time.time() - t0
    print(f"\n完成：{global_idx} 篇，索引 {OUT_INDEX}，耗时 {el/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
