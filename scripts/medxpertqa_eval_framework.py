"""MedXpertQA RAG 方法对比评测 · 统一评测框架。

按《MedXpertQA_RAG方法评测计划书》实现 6 条方法：
- M0 Baseline（无检索）
- M1 标准 RAG（BM25 检索 top-k）
- M2 推理查询扩展（先推理再检索）
- M3 思维轨迹检索（T³ 医学化，离线轨迹库）
- M4 搜索增强推理（ReAct 式，agentic）
- M5 证据过滤 + 查询重构

统一：同一 LLM（qwen-plus）、同一温度 0、同一答案格式 `Answer: <LETTER>`。
防泄漏：检索 query 只用题干主体（去选项），轨迹库与评测集无交集。

本文件是共享框架 + 检索器封装，各方法脚本 import 它。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from typing import List, Optional

sys.path.insert(0, ".")

from openai import OpenAI

# ---- 配置 ----
DATA_200 = "data/medxpertqa_p0_200.jsonl"
BM25_INDEX = "kg/medrag_pubmed/bm25_index.pkl"
BM25_META = "kg/medrag_pubmed/bm25_meta.jsonl"

# 计划书 §3：主评测 LLM 用强模型
DEFAULT_MODEL = "qwen-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 统一答题格式（计划书附 A）
FORMAT_INSTRUCTION = (
    "Output your answer in exactly this format:\n"
    "Answer: <LETTER>\n"
    "Reason: <one-sentence justification>"
)


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def get_client(model: str = DEFAULT_MODEL):
    load_env()
    return OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL), model


def call_llm(client, model, prompt, max_tokens=512, temperature=0.0):
    """带重试的 LLM 调用，返回 (text, usage)。"""
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens)
            usage = getattr(r, "usage", None)
            tok = {"prompt": getattr(usage, "prompt_tokens", 0),
                   "completion": getattr(usage, "completion_tokens", 0)} if usage else {"prompt": 0, "completion": 0}
            return r.choices[0].message.content.strip(), tok
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return "", {"prompt": 0, "completion": 0}


def parse_answer(text: str) -> str:
    """从回答中提取 `Answer: X` 或末尾的字母答案（A-J）。"""
    # 优先匹配 Answer: X
    m = re.search(r"Answer\s*:\s*([A-Ja-j])", text)
    if m:
        return m.group(1).upper()
    # 兜底：从后往前找字母
    for ch in reversed(text.upper()):
        if ch in "ABCDEFGHIJ":
            return ch
    return "?"


def build_options_str(options: dict) -> str:
    return "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))


# ---- BM25 检索器 ----
class BM25Retriever:
    """稀疏 BM25 检索器（计划书 §6 默认检索器）。"""

    def __init__(self):
        print("[BM25] 加载索引...", flush=True)
        with open(BM25_INDEX, "rb") as f:
            self.bm25 = pickle.load(f)
        self.meta = [json.loads(l) for l in open(BM25_META, encoding="utf-8")]
        print(f"[BM25] 语料 {self.bm25.corpus_size} 篇", flush=True)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 与 build_bm25_index.py 建索引时的分词保持一致：lower().split()
        return text.lower().split()

    def retrieve(self, query: str, k: int = 3) -> List[dict]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        results = []
        for i in top:
            m = self.meta[i]
            results.append({
                "title": m.get("title", ""),
                "content": m.get("content", ""),
                "score": round(float(scores[i]), 3),
            })
        return results

    @staticmethod
    def format_passages(results: List[dict], max_chars: int = 2000) -> str:
        parts = []
        for j, r in enumerate(results, 1):
            parts.append(f"[{j}] {r['title']}. {r['content']}")
        return "\n".join(parts)[:max_chars]


# ---- 评测结果落盘 ----
def save_results(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 自测：检索器 + 答案抽取
    r = BM25Retriever()
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]
    q = rows[0]["query"]
    print("检索测试:", q[:60])
    for p in r.retrieve(q, k=3):
        print(f"  [{p['score']}] {p['title'][:60]}")
    print("答案抽取测试:", parse_answer("Answer: H"), parse_answer("Reason: xxx. H"))
