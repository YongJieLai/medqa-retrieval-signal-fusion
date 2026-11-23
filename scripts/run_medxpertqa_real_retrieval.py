"""MedXpertQA 真实检索增益测量：用 PrimeKG 客观三元组（不含答案线索）做开卷。

背景：
- 旧实验 run_medxpertqa_openclose.py 的"开卷证据"是"换源 rationale"（让另一个模型
  对着正确答案生成解释），本质半泄漏 → 开卷 92% 是虚高上界。
- 本脚本用 PrimeKG（独立于答题模型的真实知识图谱）客观三元组做开卷证据，
  严格去选项/去答案/去 rationale，诚实测量检索增益 should_retrieve。

证据构造（全程不接触 label）：
  1. 从题干文本抽医学实体（body_system 映射 + 医学术语词表匹配）。
  2. 归一化字符串匹配锚定到 PrimeKG 疾病节点。
  3. 抽取疾病 1-hop 邻域三元组（disease→phenotype/drug/anatomy/disease）。
  4. 格式化为证据文本（不含选项、不含答案）。

评测：闭卷 vs 开卷（真实 KG 证据），Δ = should_retrieve。

输出：results/medxpertqa_real_retrieval.jsonl
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time

sys.path.insert(0, ".")

import pandas as pd
from openai import OpenAI

KG_PATH = "kg/primekg/kg.csv"
DATA_PATH = "data/medxpertqa_text.jsonl"
OUT = "results/medxpertqa_real_retrieval.jsonl"

# 关心临床可解释边（排除纯分子生物学边）
CLINICAL_RELATIONS = {
    "disease_disease", "disease_phenotype_positive", "disease_phenotype_negative",
    "disease_protein", "drug_disease", "disease_drug", "disease_anatomy",
    "exposure_disease", "disease_pathway", "disease_go", "disease_effect",
    "disease_mondo", "indication", "contraindication", "off-label use",
    "phenotype_present", "phenotype_absent", "manifestation", "complication",
}

# body_system -> 该系统的常见医学术语（用于从题干抽实体）
BODY_TERMS = {
    "Nervous": ["migraine", "epilepsy", "seizure", "parkinson", "alzheimer",
                "neuropathy", "meningitis", "multiple sclerosis", "stroke",
                "aneurysm", "tumor", "neuralgia", "neuro", "spinal", "nerve",
                "pupil", "palsy", "ataxia", "dementia", "tremor", "myasthenia"],
    "Cardiovascular": ["hypertension", "heart failure", "myocardial", "arrhythmia",
                       "atrial fibrillation", "coronary", "angina", "valve",
                       "aortic", "cardiomyopathy", "thrombosis", "embolism",
                       "endocarditis", "pericarditis", "tachycardia", "bradycardia"],
    "Digestive": ["gastro", "hepatitis", "cirrhosis", "pancreatitis", "cholecyst",
                  "ulcer", "gastritis", "colitis", "crohn", "diverticul",
                  "appendicitis", "reflux", "esophag", "liver", "jaundice", "bowel"],
    "Respiratory": ["pneumonia", "asthma", "copd", "bronch", "tuberculosis",
                    "pulmonary", "embolism", "pleural", "pneumothorax",
                    "fibrosis", "emphysema", "sarcoidosis", "hypox"],
    "Endocrine": ["diabetes", "thyroid", "hyperthyroid", "hypothyroid",
                  "cushing", "addison", "adrenal", "pituitary", "acromegaly",
                  "insulin", "glucose", "hormone", "parathyroid", "osteoporosis"],
    "Urinary": ["nephro", "renal", "kidney", "glomerul", "cystitis", "pyelonephritis",
                "urinary", "urolith", "hydronephrosis", "proteinuria"],
    "Reproductive": ["ovarian", "uterine", "endometri", "cervical", "prostate",
                     "testicular", "pregnancy", "eclampsia", "menstrual", "gonad"],
    "Skeletal": ["fracture", "osteo", "arthritis", "rheumatoid", "gout",
                 "scoliosis", "glenoid", "arthroplasty", "tendon", "ligament",
                 "cartilage", "lupus", "spondyl", "joint"],
    "Muscular": ["myopathy", "dystrophy", "myositis", "rhabdomyolysis",
                 "muscle", "fibromyalgia", "myasthenia", "atrophy"],
    "Lymphatic": ["lymphoma", "leukemia", "myeloma", "anemia", "spleen",
                  "lymph", "hiv", "immuno", "sickle", "thalassemia", "hemophilia"],
    "Integumentary": ["dermatitis", "psoriasis", "melanoma", "eczema", "skin",
                      "rash", "scleroderma", "vitiligo", "cellulitis", "basal cell"],
}

# 常见疾病名（跨系统，用于补充匹配）
GENERIC_DISEASE_TERMS = [
    "cancer", "carcinoma", "adenoma", "sarcoma", "melanoma", "lymphoma", "leukemia",
    "syndrome", "disease", "disorder", "deficiency", "infection", "inflammation",
]


def norm(s: str) -> str:
    """归一化：小写、去括号、去非字母数字。"""
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_disease_edges():
    """加载 PrimeKG disease 相关边 + 构建疾病名索引（内存友好）。"""
    print("加载 PrimeKG disease 边（约 981MB，需 1-2 分钟）...", flush=True)
    cols = ["relation", "display_relation", "x_type", "x_name", "y_type", "y_name"]
    df = pd.read_csv(KG_PATH, usecols=cols, low_memory=False)
    mask = (df["x_type"] == "disease") | (df["y_type"] == "disease")
    df = df[mask]
    print(f"disease 相关边数: {len(df)}", flush=True)

    # 疾病名索引
    names = set(df["x_name"][df["x_type"] == "disease"])
    names |= set(df["y_name"][df["y_type"] == "disease"])
    idx = {}
    for n in names:
        if isinstance(n, str) and n:
            idx.setdefault(norm(n), n)
    print(f"疾病节点数: {len(idx)}", flush=True)
    return df, idx


def extract_entities(question: str, body_system: str) -> list:
    """从题干文本抽医学实体（简单词表 + 正则匹配）。"""
    q_lower = question.lower()
    hits = []

    # 1. body_system 词表匹配（优先）
    terms = BODY_TERMS.get(body_system, [])
    for t in terms:
        # 单词边界匹配（前后都要求边界，避免前缀/后缀误截断）
        if re.search(r"\b" + re.escape(t) + r"\b", q_lower):
            hits.append(t)

    # 2. 通用疾病词：匹配 "xxx disease/syndrome/disorder/cancer" 模式
    for m in re.finditer(r"([a-z][a-z \-']{2,40}?)\s+(disease|syndrome|disorder|cancer|carcinoma|deficiency|anemia|infection)", q_lower):
        phrase = m.group(0).strip()
        if 3 <= len(phrase) <= 50:
            hits.append(phrase)

    # 去重保序
    seen = set()
    out = []
    for h in hits:
        h = h.strip()
        if h and norm(h) not in seen:
            seen.add(norm(h))
            out.append(h)
    return out


def anchor_entities(entities: list, disease_index: dict) -> list:
    """把抽到的实体锚定到 PrimeKG 疾病节点。返回命中的疾病名列表。"""
    hits = []
    for e in entities:
        ne = norm(e)
        if len(ne) < 4:
            continue
        # 精确匹配
        if ne in disease_index:
            hits.append((disease_index[ne], 0))
            continue
        # 子串匹配（双向，限制长度避免噪声）
        for dn, orig in disease_index.items():
            if len(dn) < 6:
                continue
            if ne in dn or dn in ne:
                hits.append((orig, 1))
                break
    seen = set()
    out = []
    for name, _ in sorted(hits, key=lambda x: x[1]):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out[:3]


def extract_neighborhood(disease_edges, disease_name: str, k: int = 5) -> list:
    """抽取疾病节点 1-hop 邻域三元组。"""
    rows = []
    sub = disease_edges[(disease_edges["x_name"] == disease_name) & (disease_edges["x_type"] == "disease")]
    for _, r in sub.iterrows():
        rows.append({"head": r["x_name"], "relation": r["display_relation"], "tail": r["y_name"], "tail_type": r["y_type"]})
    sub2 = disease_edges[(disease_edges["y_name"] == disease_name) & (disease_edges["y_type"] == "disease")]
    for _, r in sub2.iterrows():
        rows.append({"head": r["x_name"], "relation": r["display_relation"], "tail": r["y_name"], "tail_type": r["x_type"]})

    def rank(row):
        tt = str(row["tail_type"]).lower()
        if "phenotype" in tt or "drug" in tt or "anatomy" in tt:
            return 0
        if "disease" in tt:
            return 2
        return 1
    rows.sort(key=rank)
    return rows[:k]


def format_evidence(disease_name: str, triples: list) -> str:
    if not triples:
        return f"{disease_name}：PrimeKG 中未检索到足够的临床关联证据。"
    parts = [f"{t['head']} {t['relation']} {t['tail']}" for t in triples]
    return f"{disease_name} 相关知识：{'；'.join(parts)}"


def answer(client, model, question, options, evidence=None, max_tokens=10):
    opts_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    if evidence:
        p = (f"你是医学多选题助手。请结合下面给出的医学知识，选出唯一正确答案。\n"
             f"严格只输出一个字母（A/B/C/D/E/F/G/H/I/J），不要输出任何解释。\n\n"
             f"题干：{question}\n\n选项：\n{opts_lines}\n\n相关知识：\n{evidence}\n\n答案：")
    else:
        p = (f"你是医学多选题助手。请只根据题干和选项选出唯一正确答案。\n"
             f"严格只输出一个字母（A/B/C/D/E/F/G/H/I/J），不要输出任何解释。\n\n"
             f"题干：{question}\n\n选项：\n{opts_lines}\n\n答案：")
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": p}],
                temperature=0.0, max_tokens=max_tokens)
            t = r.choices[0].message.content.strip().upper()
            for ch in t:
                if ch in "ABCDEFGHIJ":
                    return ch
            return "?"
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return "?"


def main():
    load_env()
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 加载 PrimeKG
    disease_edges, disease_index = load_disease_edges()

    rows = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]
    rng = random.Random(7)
    reasoning = [r for r in rows if r["question_type"] == "Reasoning"]
    understanding = [r for r in rows if r["question_type"] == "Understanding"]
    sample = rng.sample(reasoning, 20) + rng.sample(understanding, 20)

    out = []
    stats = {"closed_ok": 0, "open_ok": 0, "retrieve": 0, "n": 0, "anchored": 0}
    t0 = time.time()

    for i, r in enumerate(sample):
        qid = r["id"]
        ans = r["label"]
        qt = r["question_type"]
        mt = r["medical_task"]
        bs = r.get("body_system", "")

        # 闭卷
        closed = answer(client, "qwen-plus", r["question"], r["options"])
        cc = int(closed == ans)

        # 真实证据构造（不含答案）
        entities = extract_entities(r["question"], bs)
        diseases = anchor_entities(entities, disease_index)
        evidence_parts = []
        for d in diseases:
            triples = extract_neighborhood(disease_edges, d, k=5)
            evidence_parts.append(format_evidence(d, triples))
        evidence = "\n".join(evidence_parts) if evidence_parts else ""

        if diseases:
            stats["anchored"] += 1

        # 开卷（真实 KG 证据）
        if evidence:
            open_a = answer(client, "qwen-plus", r["question"], r["options"], evidence)
            co = int(open_a == ans)
        else:
            open_a = closed
            co = cc

        delta = max(0, co - cc)
        should_retrieve = 1 if delta == 1 else 0

        stats["closed_ok"] += cc
        stats["open_ok"] += co
        stats["retrieve"] += should_retrieve
        stats["n"] += 1

        out.append({
            "id": qid, "question_type": qt, "medical_task": mt, "body_system": bs,
            "answer": ans, "closed_pred": closed, "open_pred": open_a,
            "correct_closed": cc, "correct_open": co, "delta": delta,
            "should_retrieve": should_retrieve,
            "entities": entities, "anchored_diseases": diseases,
            "evidence": evidence, "evidence_len": len(evidence),
        })

        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(sample)}] 闭卷对={stats['closed_ok']} 开卷对={stats['open_ok']} "
                  f"救回={stats['retrieve']} 锚定={stats['anchored']} 耗时{round(el,0)}s", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = stats["n"]
    print(f"\n=== 汇总（真实 PrimeKG 证据，qwen-plus）===", flush=True)
    print(f"闭卷: {stats['closed_ok']}/{n} = {stats['closed_ok']/n:.0%}", flush=True)
    print(f"开卷: {stats['open_ok']}/{n} = {stats['open_ok']/n:.0%}", flush=True)
    print(f"should_retrieve=1: {stats['retrieve']}/{n}", flush=True)
    print(f"锚定到疾病: {stats['anchored']}/{n}", flush=True)
    print(f"\n已保存到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
