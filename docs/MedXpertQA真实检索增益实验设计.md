# MedXpertQA 真实检索增益实验设计

> 面向「MedXpertQA 类专家级、防泄漏、10 选 1」基准 · 整理于 2026-11-22
> 目的：用**真实、不含答案线索**的知识源，诚实测量检索增益（should_retrieve），验证「融合 > 单独」的充分性证据。

---

## 0. 背景与问题定位

### 0.1 为什么之前的测量不可信

上一轮 `run_medxpertqa_openclose.py` 的"开卷证据"设计是：**让另一个模型对着正确答案生成 rationale，再喂给被评测模型**（换源 rationale）。

| 模型 | 闭卷 | 开卷 | should_retrieve |
|---|---|---|---|
| qwen-plus（解释来自 flash） | 10% | **92%** | 33/40 |
| qwen-flash（解释来自 plus） | 18% | 82% | 26/40 |

这是所有数据集中第一次 should_retrieve 显著 > 0，**但开卷 92% 不是真实水平，而是"对着正确答案生成解释"的半泄漏上界**：

- rationale 是"为什么正确答案是对的"，直接锚定了正确选项，等于变相喂答案；
- 换源（flash↔plus）只能缓解"同模型自循环"，不能消除"答案已被编进解释"的泄漏。

### 0.2 调研笔记给出的正确做法（四条医学专用化）

对 MedXpertQA 这类专家级基准，有效检索的关键是四个环节都医学专用化：

| 环节 | 结论 |
|---|---|
| ① 语料库 | PubMed 不够，要叠教科书 + 指南 + 知识图谱（Omni-RAG 的 MedOmniKB） |
| ② 检索器 | 医学专用 dense（MedCPT，PubMed 点击日志对比学习） |
| ③ Query 构造 | **去选项防泄漏** + rationale 改写 |
| ④ 检索后 | 证据过滤 + rerank（top-16 只有 22% 相关，噪声反而伤模型） |

**最反直觉的一条**：医学 RAG 瓶颈不在"检索不够强"，而在"检索回来的噪声伤害本来就会的模型"——只有 22% 段落相关，78% 是垃圾，直接塞给 GPT-4o 反而把 factuality 带掉 6%。

### 0.3 本实验要回答的问题

1. **去掉泄漏后，真实检索增益还有多少？** should_retrieve 是否从 33/40 塌缩回接近 0？
2. **真实证据能否救回"鉴别诊断/临床判断"类错题？** 还是只能救"缺知识"类错题？
3. **MedXpertQA 是否本就"不需要检索"？**（题干已含全部信息，错题归因是判断而非知识缺口）

---

## 1. 防泄漏设计（核心）

### 1.1 证据来源：PrimeKG（真实知识图谱，独立于答题模型）

- 源：`kg/primekg/kg.csv`（810 万行，17,050 疾病节点，关系三元组）。
- 性质：**独立于千问**的第三方医学知识图谱（整合 20+ 公开生物医学数据库），不含任何题目答案线索。
- 与旧实验的对比：旧实验证据 = 模型自生成（泄漏）；本实验证据 = 客观 KG 三元组（无泄漏）。

### 1.2 证据构造（去选项、去答案、去 rationale）

对每题，严格按以下顺序构造证据，**全程不接触 label 字段**：

1. **从题干文本抽医学实体**：MedXpertQA 没有 `metamap_phrases` 字段（旧脚本依赖它），需从 `question` 文本抽取疾病/症状/药物实体。
   - 方案 A（首选）：用题干中的 `body_system` + 词表匹配（MedXpertQA 自带 body_system 12 类）。
   - 方案 B：简易医学名词抽取（正则 + 高频医学术语词表）。
2. **锚定到 PrimeKG 疾病节点**：归一化字符串匹配（复用 `extract_kg_evidence.py` 的 `norm()` + `anchor_entities()` 逻辑）。
3. **抽取 1-hop 邻域三元组**：疾病 → phenotype/drug/anatomy/disease 关系，取临床可解释边。
4. **格式化为证据文本**：`疾病名 相关知识：head relation tail；...`（不含选项、不含答案、不含"为什么对"）。

### 1.3 关键约束（防泄漏红线）

- ❌ 不使用 `label` / 正确选项构造证据（旧实验的泄漏源）。
- ❌ 不用"为什么答案 X 对"的 rationale（答案锚定）。
- ❌ 不做"去选项"只针对题干主体（题干本身含信息，但检索 query 只用题干关键实体）。
- ✅ 证据只来自 PrimeKG 客观三元组。

---

## 2. 实验设置

### 2.1 数据

- **MedXpertQA** 2450 题，10 选 1（A-J），question_type = Reasoning(1861) / Understanding(589)，medical_task = Diagnosis/Treatment/Basic Science，body_system 12 类。
- 抽样：40 题（Reasoning 20 + Understanding 20），与旧实验同 seed（random.Random(7)）保证可比。

### 2.2 Backbone

- 主模型：qwen-plus（与旧实验对齐）。
- 辅助：qwen-flash（弱 backbone，验证结论稳健性）。

### 2.3 三臂评测（每臂独立跑）

| 臂 | 证据 | 用途 |
|---|---|---|
| **闭卷** | 无 | baseline，测模型参数记忆 |
| **开卷-真实** | PrimeKG 三元组（去选项/去答案） | **本实验核心**，测真实增益 |
| （对照）开卷-泄漏 | 换源 rationale（旧实验） | 定量展示泄漏造成的虚高 |

### 2.4 指标

- `correct_closed` / `correct_open`：闭卷/开卷正确率。
- `delta = max(0, correct_open - correct_closed)`。
- `should_retrieve = 1 if delta == 1 else 0`（检索增益 ground truth）。
- 分题型统计：Reasoning vs Understanding × Diagnosis/Treatment/Basic Science。

---

## 3. 可证伪预测

- **预测 1（主）**：真实证据下 `should_retrieve` 从 33/40 显著塌缩（预期 < 5/40），证明旧 92% 是泄漏虚高。
- **预测 2**：真实证据只能救回"缺知识"类错题，救不回"鉴别诊断/临床判断"类错题（选项差异细微，KG 三元组不含足够判别信息）。
- **预测 3**：开卷-真实正确率 ≈ 闭卷（甚至略低，若 KG 证据答非所问），因为 MedXpertQA 题干已含全部信息。

---

## 4. 对标基线（related work 对齐）

- **Rethinking RAG**：GPT-4o + RAG（MedCPT k=16 + 证据过滤）在 MedXpertQA 上 **35.4%**；标准 RAG 只有 22% 段落相关。
- **MedCPT**（Jin et al. 2023）：医学专用 dense 检索器，2.55 亿 PubMed 点击日志对比学习。
- **Omni-RAG / MedOmniKB**：Books 27.7k + Guidelines 45.7k + PubMed 25.3M + KG（UMLS/DrugBank）。

> 本实验用 PrimeKG 作为 KG 证据源（UMLS 无 license 的退路），是对标 Omni-RAG 的 KG 组件；若后续可接 MedCPT 检索，可补 PubMed 语料 IDF + 真实检索链路。

---

## 5. 对本研究主叙事的意义

无论本实验 should_retrieve 结果如何，结论都指向同一处：

- **若 should_retrieve 塌缩**：坐实「医学 MCQ 检索增益本质极低」+「旧 92% 是泄漏」，进一步强化「coverage 方法的正确性叙事应建立在**信号正交互补**（Spearman ρ<0.15、并集兜错题 10>7）这一不依赖检索增益标签的稳健基础上」。
- **若 should_retrieve 仍 > 0**：真实证据确实能带来增益，则「融合 > 单独」的充分性证据成立，可对标 35.4% 基线推进三臂主实验。

---

## 6. 实施步骤

1. 写 `scripts/run_medxpertqa_real_retrieval.py`：
   - 从 MedXpertQA 题干抽实体 → 锚定 PrimeKG → 抽邻域三元组 → 构造证据。
   - 闭卷 vs 开卷-真实 vs 开卷-泄漏（复用旧结果）三臂。
2. 跑 40 题（seed=7，与旧实验一致）。
3. 输出 `results/medxpertqa_real_retrieval.jsonl`。
4. 分析错题归因，与旧泄漏版对比，输出结论。

---

## 7. 风险与边界

- **实体锚定噪声**：MedXpertQA 题干是完整临床场景，无 meta 标注，从题干抽实体可能锚不准核心疾病（旧实验在 MedQA hard 上已暴露此问题）。缓解：先抽 body_system 对应高频词，再锚定。
- **PrimeKG 三元组无判别性**：KG 只给"疾病有哪些症状/药物"，不给"这道题的选项差异"，可能救不回细微判别题。这是**诚实的边界**，不是失败。
- **千问额度**：qwen-plus/flash 可用，qwen3.5-27b 太慢已停用。
