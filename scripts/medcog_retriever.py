"""MedCoG 复现 · 知识执行器（Knowledge Executor）的检索组件。

提供两类知识检索：
1. Factual Knowledge（KG 验证）：BM25 检索 PubMed 语料 + PrimeKG 三元组查询
   - MedCoG 原论文用 PrimeKG 做 query-hypothesis 验证（最短路径 → top-K）
   - 本复现：BM25 检索 PubMed 摘要（本地 199094 篇）+ PrimeKG 三元组（kg/primekg/kg.csv）

2. Episodic Knowledge（情景记忆）：case bank 相似题检索
   - 用 MedXpertQA 全集（非 P0 200 题部分）作案例库
   - BM25 检索相似题干，返回历史案例（题干+答案，供 SCoT 参考）

防泄漏：检索 query 只用题干主体，不碰 label；memory 排除当前题。

可被 run_medcog.py import。
"""
from __future__ import annotations

import json
import os
import pickle
import re
from typing import List, Optional

import numpy as np

BM25_INDEX = "kg/medrag_pubmed/bm25_index.pkl"
BM25_META = "kg/medrag_pubmed/bm25_meta.jsonl"
PRIMEKG_CSV = "kg/primekg/kg.csv"
MEDXPERTQA_FULL = "data/medxpertqa_text.jsonl"
P0_200 = "data/medxpertqa_p0_200.jsonl"


def _tokenize(text: str) -> List[str]:
    """简单英文分词（与建索引时的分词保持一致即可）。"""
    return re.findall(r"[a-z0-9]+", text.lower())


class MedCoGRetriever:
    """MedCoG 知识检索器：Factual（BM25 PubMed + PrimeKG）+ Episodic（case bank）。"""

    def __init__(self, build_case_bank: bool = True):
        # BM25 PubMed 语料
        print("[MedCoGRetriever] 加载 BM25 索引...", flush=True)
        with open(BM25_INDEX, "rb") as f:
            self.bm25 = pickle.load(f)
        self.meta = [json.loads(l) for l in open(BM25_META, encoding="utf-8")]
        print(f"  BM25 语料 {self.bm25.corpus_size} 篇，meta {len(self.meta)} 条", flush=True)

        # PrimeKG 三元组（懒加载：只建 disease 名称 → 三元组索引）
        self._primekg_loaded = False
        self._disease_triples = {}

        # Episodic case bank
        self.case_bank = []
        self._case_tokens = []
        if build_case_bank:
            self._build_case_bank()

    # ---------- Factual：BM25 PubMed ----------
    def retrieve_pubmed(self, query: str, k: int = 8) -> List[dict]:
        """BM25 检索 PubMed 摘要，返回 top-k 标题+内容。"""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        results = []
        for i in top:
            m = self.meta[i]
            results.append({"title": m.get("title", ""), "content": m.get("content", ""), "score": round(float(scores[i]), 3)})
        return results

    @staticmethod
    def format_pubmed(results: List[dict], max_chars: int = 1500) -> str:
        """格式化 PubMed 检索结果。"""
        parts = []
        for j, r in enumerate(results, 1):
            parts.append(f"[{j}] {r['title']}. {r['content']}")
        text = "\n".join(parts)
        return text[:max_chars]

    # ---------- Factual：PrimeKG ----------
    def _load_primekg(self):
        if self._primekg_loaded:
            return
        print("[MedCoGRetriever] 加载 PrimeKG 三元组...", flush=True)
        disease_relations = {
            "disease_phenotype_positive", "disease_phenotype_negative",
            "disease_protein", "disease_disease", "drug_drug",
        }
        # 只索引 disease 相关的医学关系（缩小内存）
        import csv
        count = 0
        with open(PRIMEKG_CSV, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 11:
                    continue
                relation = row[0]
                if relation not in disease_relations:
                    continue
                x_type, x_name = row[4], row[5]
                y_type, y_name = row[9], row[10]
                # 三元组：subject - relation - object
                for subj, st, obj, ot in [(x_name, x_type, y_name, y_type), (y_name, y_type, x_name, x_type)]:
                    key = subj.lower()
                    self._disease_triples.setdefault(key, []).append(
                        {"relation": relation, "subject": subj, "object": obj}
                    )
                count += 1
        self._primekg_loaded = True
        print(f"  PrimeKG 索引 {count} 条疾病相关三元组，{len(self._disease_triples)} 个实体", flush=True)

    def retrieve_primekg(self, entities: List[str], max_per_entity: int = 3) -> List[dict]:
        """给定实体名列表，检索 PrimeKG 三元组。"""
        self._load_primekg()
        results = []
        seen = set()
        for e in entities:
            triples = self._disease_triples.get(e.lower(), [])
            for t in triples[:max_per_entity]:
                key = (t["relation"], t["subject"], t["object"])
                if key in seen:
                    continue
                seen.add(key)
                results.append(t)
        return results

    @staticmethod
    def format_primekg(results: List[dict], max_items: int = 10) -> str:
        """格式化 PrimeKG 三元组为自然语言。"""
        parts = []
        for r in results[:max_items]:
            parts.append(f"{r['subject']} --[{r['relation']}]--> {r['object']}")
        return "\n".join(parts) if parts else "（未检索到相关知识图谱三元组）"

    # ---------- Episodic：case bank ----------
    def _build_case_bank(self):
        """用 MedXpertQA 全集（排除 P0 200 题）作案例库。"""
        p0_ids = {json.loads(l)["id"] for l in open(P0_200, encoding="utf-8")}
        self.case_bank = []
        for l in open(MEDXPERTQA_FULL, encoding="utf-8"):
            r = json.loads(l)
            if r.get("id") in p0_ids:
                continue
            self.case_bank.append(r)
        self._case_tokens = [_tokenize(r.get("question", "")) for r in self.case_bank]
        print(f"[MedCoGRetriever] case bank 构建完成：{len(self.case_bank)} 个历史案例", flush=True)

    def retrieve_cases(self, query: str, k: int = 3) -> List[dict]:
        """检索相似历史案例（按题干 BM25 相似度）。返回案例（题干+正确答案标签，不含解释）。"""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        # 简单 BM25 相似度（复用已加载的 bm25 会对 PubMed 打分，这里单独算 Jaccard/BM25）
        scores = []
        q_set = set(q_tokens)
        for toks in self._case_tokens:
            t_set = set(toks)
            if not t_set:
                scores.append(0.0)
                continue
            inter = len(q_set & t_set)
            union = len(q_set | t_set)
            scores.append(inter / union if union else 0.0)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.case_bank[i] for i in top if scores[i] > 0]

    @staticmethod
    def format_cases(cases: List[dict], max_chars: int = 800) -> str:
        """格式化历史案例（题干 + 正确答案，供 SCoT 参考）。"""
        if not cases:
            return "（无相似历史案例）"
        parts = []
        for j, c in enumerate(cases, 1):
            q = c.get("question", "")[:200]
            label = c.get("label", "?")
            parts.append(f"案例{j}：{q}... [正确答案: {label}]")
        return "\n".join(parts)[:max_chars]


if __name__ == "__main__":
    # 自测
    r = MedCoGRetriever()
    q = "A 37-year-old male seeks medical attention for progressive difficulty swallowing"
    print("\n=== PubMed 检索 ===")
    for p in r.retrieve_pubmed(q, k=3):
        print(f"  [{p['score']}] {p['title'][:80]}")
    print("\n=== PrimeKG 检索 ===")
    for t in r.retrieve_primekg(["hypertension", "dysphagia"]):
        print(f"  {t['subject']} --[{t['relation']}]--> {t['object']}")
    print("\n=== Case bank 检索 ===")
    for c in r.retrieve_cases(q, k=2):
        print(f"  {c.get('id')}: {c.get('question','')[:80]} [答案:{c.get('label')}]")
