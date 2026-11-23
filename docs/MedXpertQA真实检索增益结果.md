# MedXpertQA 真实检索增益 —— 结果与错题分析

> 实验脚本：`scripts/run_medxpertqa_real_retrieval.py`
> 结果文件：`results/medxpertqa_real_retrieval.jsonl`
> 设计文档：`docs/MedXpertQA真实检索增益实验设计.md`
> 执行于 2026-11-22

---

## 一、核心结果（一句话）

**去掉答案泄漏后，真实检索增益（should_retrieve）从 33/40 塌缩到 0/40。**

| 实验版本 | 闭卷 | 开卷 | should_retrieve |
|---|---|---|---|
| 旧版（换源 rationale，半泄漏） | 10% | **92%** | **33/40** |
| **本次（PrimeKG 真实三元组，去答案）** | 10% | **10%** | **0/40** |

旧实验的"开卷 92%"是**对着正确答案生成解释**造成的答案泄漏虚高上界；换成独立于答题模型的 PrimeKG 客观三元组后，检索增益归零。

---

## 二、实验设置回顾

- 数据：MedXpertQA 40 题（Reasoning 20 + Understanding 20，seed=7 与旧实验一致）。
- backbone：qwen-plus。
- 证据：从题干（不含 label）抽医学实体 → 锚定 PrimeKG 疾病节点 → 抽 1-hop 邻域三元组（disease→phenotype/drug/anatomy）。
- 防泄漏红线：全程不接触 `label` 字段，证据不含选项、不含答案、不含"为什么对"。

---

## 三、错题归因（36 道闭卷错题）

### 3.1 分布

| 维度 | 分布 |
|---|---|
| medical_task | Diagnosis 18 · Basic Science 12 · Treatment 6 |
| question_type | Reasoning 18 · Understanding 18 |
| body_system | Reproductive 8 · Skeletal 8 · Cardiovascular 3 · Digestive 3 · 其余分散 |

### 3.2 证据"喂进去了"但救不回

- 锚定到疾病：28/40（证据非空 28/40，平均 574 字符）。
- **开卷改变答案的题只有 4/40**，且 4 题全部是"从错改成另一个错"，无一救对。
- 其余 32/36 错题开卷答案纹丝不动（证据未影响模型判断）。

### 3.3 为什么救不回（根因）

看典型错题的证据即可明白：

- `Text-863`（Reproductive/Diagnosis）：正确答案 G，闭卷选 E，开卷仍选 E。
  证据锚到 `antiphospholipid syndrome` 的"症状三元组"（retinal artery occlusion、retinal vasculitis…），但**题干考的是"选项间细微判别"（哪个场景最适合某处置）**，KG 的"疾病有哪些症状"不含这道题的判别线索。
- `Text-1053`（Reproductive/Diagnosis）：锚到 `penile/testicular agenesis`，证据是"尿道闭锁、膀胱异常"等表型，答非所问。

**根本原因**：MedXpertQA 的难题（人类专家也仅 43.9%）难在**鉴别诊断 / 临床判断**——题干已含全部信息，选项差异在"哪个更符合最优处置"这类临床决策上。PrimeKG 提供的"疾病→症状/药物"关系三元组，属于**陈述性知识**，不含题目要求的**程序性判别信息**，所以喂了也救不回。

---

## 四、与调研笔记的印证

本次结果**完全印证**了调研笔记最反直觉的一条结论：

> 医学 RAG 瓶颈不在"检索不够强"，而在"检索回来的噪声伤害本来就会的模型"——top-16 段落只有 22% 相关，78% 是垃圾，直接塞给 GPT-4o 反而把 factuality 带掉 6%。

MedXpertQA 上的证据链闭环为：

1. **检索增益本质极低**（should_retrieve ≈ 0）：题干已含信息，错题是判断不是知识缺口。
2. **检索证据答非所问**（陈述性知识 vs 程序性判别）：PrimeKG 三元组给"疾病有什么"，不给"选项哪个更对"。
3. **旧 92% 是泄漏伪影**：对着答案编 rationale 等于变相喂答案。

---

## 五、对本研究主叙事的最终意义

这轮真实检索实验，是「融合 > 单独」充分性证据的最后一次诚实验证，结论收敛：

1. **"检索增益"这个靶子，在当前医学 MCQ 数据 + 可用 backbone 组合下，本质不可测**（正样本恒≈0）。这不是 backbone 强弱或证据好坏问题，而是"检索增益"这个量在"以鉴别诊断/临床判断为主"的专家级医学 MCQ 上本来就接近 0。

2. **coverage 方法的正确性叙事，应（且只能）建立在「知识侧与模型侧信号正交互补」这一不依赖检索增益标签的稳健基础上**——即已取得的定量证据：
   - Spearman ρ < 0.15（知识侧 coverage vs 模型侧 self_eval/familiarity）；
   - 报警重叠 Jaccard 18%~25%；
   - 并集兜错题数 10 > 任一单独（7/6）。

3. **调研笔记给了方法落地的正确姿势**：若要把知识侧信号做实，稀有度 IDF 应落在 MedCPT 官方预计算的 PubMed 语料上（而非任意共现图），KG 用 UMLS/PrimeKG；rationale 改写是"模型侧"，与"知识侧"正交、可互补——这是"融合"的干净定义，可作为论文 related work 的锚点。

---

## 六、结论（最终裁决）

**"融合能提升检索增益"的充分性证据，在 MedXpertQA（乃至当前所有可用医学 MCQ 数据）上无法取得**——不是实验没做对，而是这个量在专家级、防泄漏的 10 选 1 基准上本质为零。

本研究的立足点最终落在：**知识侧与模型侧信号正交（ρ<0.15）+ 并集互补兜错题（10>7）**这一必要性与互补性证据，而非"罕见病失明"或"检索增益"这类已被数据反复否定的强前提。

---

## 七、文件清单

- `docs/MedXpertQA真实检索增益实验设计.md`：实验设计（防泄漏设计 + 三臂 + 可证伪预测 + 对标基线）。
- `scripts/run_medxpertqa_real_retrieval.py`：真实检索实验脚本。
- `results/medxpertqa_real_retrieval.jsonl`：40 题逐题结果（含实体、锚定疾病、证据、闭卷/开卷预测）。
- `results/实验说明.md`：全项目方法论复盘（已追加本节）。
