"""从 PrimeKG 抽取疾病实体的真实 KG 三元组，作为开卷检索证据。

背景：MVP 里"开卷证据"用的是 qwen-plus 自生成解释（DEVIATION #2），
这是模型自循环，不是真实检索。本脚本用 PrimeKG（独立于答题模型的
真实知识图谱）抽取疾病节点的邻域三元组，替换自生成解释。

步骤：
1. 加载 PrimeKG kg.csv（0.94GB），筛选 disease 相关边。
2. 对 MedQA hard 每题的 metamap_phrases，清洗后锚定到 PrimeKG 疾病节点。
3. 抽取疾病节点的 1-hop 邻域三元组（disease→drug/phenotype/anatomy/...）。
4. 落成 results/kg_evidence.jsonl，作为开卷 evidence 字段。

注意：
- kg.csv 主文件只有关系边（12 列），疾病文本描述在 feature 文件里
  （mayo.csv / orpha.csv），本脚本先用关系三元组作为证据（也是真实知识）。
- 锚定用归一化字符串匹配（去空格/小写/下划线），不做向量检索（MVP）。
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, ".")

import pandas as pd

KG_PATH = "kg/primekg/kg.csv"
MEDQA_HARD = "data/medqa_hard.jsonl"
OUT = "results/kg_evidence.jsonl"

# 我们关心的边类型（临床可解释的，排除纯分子生物学边）
CLINICAL_RELATIONS = {
    "disease_disease",
    "disease_phenotype_positive",
    "disease_phenotype_negative",
    "disease_protein",
    "drug_disease",
    "disease_drug",
    "disease_anatomy",
    "exposure_disease",
    "disease_pathway",
    "disease_go",
    "disease_effect",
    "disease_mondo",
    "indication",
    "contraindication",
    "off-label use",
    "phenotype_present",
    "phenotype_absent",
    "manifestation",
    "complication",
}

# 归一化：去空格/下划线/连字符，统一小写，去括号内容
def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def load_disease_edges() -> pd.DataFrame:
    """加载 kg.csv 并筛出 disease 相关边（内存友好：只读需要的列）。"""
    cols = ["relation", "display_relation", "x_type", "x_name",
            "y_type", "y_name"]
    df = pd.read_csv(KG_PATH, usecols=cols, low_memory=False)
    # 只保留至少一端是 disease 的边
    mask = (df["x_type"] == "disease") | (df["y_type"] == "disease")
    return df[mask]


def build_disease_index(disease_edges: pd.DataFrame) -> Dict[str, str]:
    """构建 归一化疾病名 -> 原始疾病名 的索引。"""
    names = set(disease_edges["x_name"][disease_edges["x_type"] == "disease"])
    names |= set(disease_edges["y_name"][disease_edges["y_type"] == "disease"])
    idx = {}
    for n in names:
        if isinstance(n, str) and n:
            idx.setdefault(norm(n), n)
    return idx


def anchor_entities(phrases: List[str], disease_index: Dict[str, str]) -> List[str]:
    """把题目的 metamap_phrases 锚定到 PrimeKG 疾病节点，返回命中的疾病名。

    改进（修复噪声锚定）：
    1. 过滤短短语（<=3 字符）和纯功能词。
    2. 优先精确 norm 匹配；子串匹配要求短语长度 >= 6 且疾病名长度 >= 6，
       避免 2 字符短语（如 "mo"）锚到无关疾病。
    3. 每题最多返回 3 个疾病。
    """
    from scripts.compute_coverage import clean_phrases

    cleaned = clean_phrases(phrases)
    hits = []
    # 第一遍：精确匹配（norm 后完全相等）
    for p in cleaned:
        np = norm(p)
        if len(np) >= 4 and np in disease_index:
            hits.append((disease_index[np], 0))  # 0 = 精确
    # 第二遍：子串匹配（双向，限制长度避免噪声）
    if len(hits) < 3:
        for p in cleaned:
            np = norm(p)
            if len(np) < 6:
                continue
            for dn, orig in disease_index.items():
                if len(dn) < 6:
                    continue
                if np in dn or dn in np:
                    hits.append((orig, 1))  # 1 = 子串
                    break
    # 去重保序（保留精确匹配优先）
    seen = set()
    out = []
    for name, _ in sorted(hits, key=lambda x: x[1]):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out[:3]


def extract_neighborhood(disease_edges: pd.DataFrame, disease_name: str, k: int = 5) -> List[Dict]:
    """抽取疾病节点的 1-hop 邻域三元组（最多 k 条，按关系类型优先）。"""
    rows = []
    # x 端是疾病
    sub = disease_edges[
        (disease_edges["x_name"] == disease_name) & (disease_edges["x_type"] == "disease")
    ]
    for _, r in sub.iterrows():
        rows.append({
            "head": r["x_name"], "relation": r["display_relation"],
            "tail": r["y_name"], "tail_type": r["y_type"],
        })
    # y 端是疾病
    sub2 = disease_edges[
        (disease_edges["y_name"] == disease_name) & (disease_edges["y_type"] == "disease")
    ]
    for _, r in sub2.iterrows():
        rows.append({
            "head": r["x_name"], "relation": r["display_relation"],
            "tail": r["y_name"], "tail_type": r["x_type"],
        })
    # 优先临床边（phenotype/drug/anatomy 相关），再取前 k
    def rank(row):
        rel = str(row["relation"]).lower()
        tt = str(row["tail_type"]).lower()
        if "phenotype" in tt or "drug" in tt or "anatomy" in tt:
            return 0
        if "disease" in tt:
            return 2
        return 1
    rows.sort(key=rank)
    return rows[:k]


def format_evidence(disease_name: str, triples: List[Dict]) -> str:
    """把三元组格式化成一句话证据文本。"""
    if not triples:
        return f"{disease_name}：PrimeKG 中未检索到足够的临床关联证据。"
    parts = []
    for t in triples:
        parts.append(f"{t['head']} {t['relation']} {t['tail']}")
    return f"{disease_name} 相关知识：{'；'.join(parts)}"


def main():
    print("加载 PrimeKG（约 0.94GB，首次需数分钟）...")
    disease_edges = load_disease_edges()
    print(f"disease 相关边数: {len(disease_edges)}")

    print("构建疾病名索引...")
    disease_index = build_disease_index(disease_edges)
    print(f"疾病节点数: {len(disease_index)}")

    gt_rows = [json.loads(l) for l in open(MEDQA_HARD, encoding="utf-8")]

    out_rows = []
    n_hit = 0
    for r in gt_rows:
        phrases = r["metamap_phrases"]
        diseases = anchor_entities(phrases, disease_index)
        evidence_parts = []
        for d in diseases[:3]:  # 每题最多 3 个疾病
            triples = extract_neighborhood(disease_edges, d, k=5)
            evidence_parts.append(format_evidence(d, triples))
        evidence = "\n".join(evidence_parts) if evidence_parts else ""
        if evidence:
            n_hit += 1
        out_rows.append({
            "id": r["id"],
            "question": r["question"],
            "answer_idx": r["answer_idx"],
            "anchored_diseases": diseases,
            "evidence": evidence,
        })

    import os
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== 摘要 ===")
    print(f"总题数: {len(out_rows)}")
    print(f"成功锚定疾病并抽取证据: {n_hit}/{len(out_rows)} ({n_hit/len(out_rows):.1%})")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
