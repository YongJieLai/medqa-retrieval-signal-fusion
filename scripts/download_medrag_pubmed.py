"""下载 MedRAG/pubmed 语料 chunk（通过 hf-mirror，替代被限速的 NCBI）。

下载前 N 个 chunk（每个约 24-36MB，含约 2.4 万篇 PubMed 摘要）。
默认 200 个 chunk ≈ 11GB ≈ 480 万篇，覆盖医学 MCQ 足够。
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

import requests

BASE = "https://hf-mirror.com/datasets/MedRAG/pubmed/resolve/main/chunk/pubmed23n{idx:04d}.jsonl"
OUT_DIR = "kg/medrag_pubmed"
N_CHUNKS = 200  # 下载前 200 个 chunk


def download(idx: int) -> tuple[int, str]:
    url = BASE.format(idx=idx)
    out = f"{OUT_DIR}/pubmed23n{idx:04d}.jsonl"
    if os.path.exists(out) and os.path.getsize(out) > 1000000:
        return idx, "skip"
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            tmp = out + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            os.rename(tmp, out)
            return idx, f"{os.path.getsize(out)/1e6:.1f}MB"
        except Exception as e:
            if attempt == 2:
                return idx, f"FAIL {e}"
            time.sleep(2)
    return idx, "FAIL"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(download, i): i for i in range(1, N_CHUNKS + 1)}
        for fut in as_completed(futs):
            idx, status = fut.result()
            done += 1
            if done % 20 == 0:
                el = time.time() - t0
                print(f"  [{done}/{N_CHUNKS}] 耗时 {round(el,0)}s", flush=True)
    el = time.time() - t0
    total = sum(os.path.getsize(f"{OUT_DIR}/{f}") for f in os.listdir(OUT_DIR) if f.endswith(".jsonl"))
    print(f"\n完成 {N_CHUNKS} chunk，总 {total/1e9:.2f}GB，耗时 {round(el,0)}s", flush=True)


if __name__ == "__main__":
    main()
