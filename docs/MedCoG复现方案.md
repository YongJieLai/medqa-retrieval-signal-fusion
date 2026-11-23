# MedCoG 在 MedXpertQA 200 题上的完整复现方案

> 目标：把 ICML 2026 的 MedCoG（医学元认知智能体）从零复现到 MedXpertQA P0 的 200 题上，用 qwen-flash 作 backbone，对比既有 P0 基线（闭卷 17.5%、RAG 全负增益），验证"元认知门控"能否突破"检索增益为负"的结论。

---

## 1. 背景：MedCoG 是什么

MedCoG（arXiv 2602.07905，ICML 2026）提出用 LLM 的**元认知**（自评自身认知状态）来调控推理过程，避免"盲目加知识"带来的噪声负迁移。它和本研究是同一战场的两个阵营：

| | 信号来源 | 调控对象 | 本文定位 |
|---|---|---|---|
| MedCoG | **模型侧元认知**（LLM 自评） | 用不用 KG/Memory/CoT | 对比对象 |
| 本研究 coverage | **知识侧**（KG+语料稀有度×必要性） | 检不检索 | 本文方法 |

MedCoG 的核心洞察与本研究的 P0 发现**完全同向**：盲目加 KG/Memory 会引入噪声负迁移（"more is less"）。

## 2. MedCoG 框架（三大组件）

### 2.1 Meta-Cognition Regulator（元认知调节器）

三个元认知维度，每个打一个 0~1 分数，构成状态向量 `s = [s_c, s_f, s_k]`：

| 维度 | 含义 | 论文定义 |
|---|---|---|
| Complexity `s_c` | 是否需要多跳推理/逻辑分解 | 结构性复杂度 |
| Familiarity `s_f` | 是否教科书典型案例（能否复用历史经验） | 熟悉度 |
| Knowledge Density `s_k` | 是否依赖具体医学事实（实体/关系） | 知识密度 |

### 2.2 非参数门控路由

`I_j = 1(s_j > τ_j)`，阈值 `τ = [τ_c, τ_f, τ_k]` 在 held-out 集校准。策略选择：

```
M = π(s; τ) = 
    Zero-Shot,            若 I_c ∧ I_f ∧ I_k = 0（三维全低）
    SCoT ⊕ Σ_{j∈{f,k}} I_j·M_j,  否则
```

策略池 `S = {Zero-Shot, SCoT, SCoT+Memory, SCoT+KG, SCoT+KG+Memory}`：

| 策略 | 含义 |
|---|---|
| Zero-Shot | 直接作答，不推理 |
| SCoT | 结构化思维链（结构知识引导的逐步推理） |
| SCoT+Memory | SCoT + 复用历史相似案例（episodic memory） |
| SCoT+KG | SCoT + 知识图谱验证（factual knowledge） |
| SCoT+KG+Memory | 全部叠加 |

### 2.3 Knowledge Executor（知识执行器）

三类知识：
- **Procedural（程序性）**：SCoT 结构知识（事实层/推理层解耦）
- **Episodic（情景性）**：历史推理案例（case bank）
- **Factual（事实性）**：KG 验证（query-hypothesis 对 → 最短路径 → top-K）

---

## 3. 落地映射：MedCoG 组件 → 本仓库资产

| MedCoG 组件 | 本仓库落地 | 说明 |
|---|---|---|
| Complexity 评分 | 新写 LLM prompt | qwen-flash 打 0~1 分 |
| Familiarity 评分 | 复用 `LLMClient.familiarity()` | logprob margin（最佳/次优 token 差距）|
| Knowledge Density 评分 | 新写 LLM prompt | qwen-flash 打 0~1 分 |
| SCoT | 新写结构化 CoT prompt | 事实层+推理层解耦 |
| KG 验证 | BM25 检索 PubMed + PrimeKG 三元组 | 本地 `kg/medrag_pubmed/bm25_index.pkl` |
| Episodic Memory | case bank + BM25 相似题检索 | 用 MedXpertQA 全集非 P0 题作案例库 |
| 阈值门控 | 留 40 题校准 τ | 分位数/网格搜索 |

---

## 4. 实验设计

### 4.1 数据切分

- **校准集**：40 题（random_state 固定，从 P0 200 题中划出，**与测试集不重叠**）
- **测试集**：160 题

### 4.2 阈值校准

在 40 题校准集上，对每个维度 τ_j 做网格搜索（如 `{0.3, 0.5, 0.7}`），目标：最大化校准集准确率（或最大化"检索增益"）。**关键防泄漏**：校准只用校准集，测试集完全独立。

### 4.3 评测指标

- 主指标：MedCoG 五策略路由后的**测试集准确率**
- 对比基线：P0 的 M0a 闭卷 17.5%、M1 RAG 全负
- 辅助：元认知信号分布（三维分数的统计）、策略路由分布、门控增益（vs 固定全 SCoT+KG+Memory）

### 4.4 防泄漏铁律

1. 检索 query 只用题干主体（去选项），全程不碰 label
2. KG 验证的 query-hypothesis 对**不包含答案字母**
3. Episodic memory 案例库**排除当前题**，且案例不喂答案
4. 阈值校准与测试集严格隔离

---

## 5. 实施步骤

1. **元认知监测**：200 题打三维分数 → `results/medcog_signals.jsonl`
2. **KG 验证 + Memory 检索**：BM25 检索器封装 + case bank 构建
3. **五策略推理**：实现 Zero-Shot/SCoT/SCoT+Mem/SCoT+KG/SCoT+KG+Mem 的 prompt 与调用
4. **阈值门控**：40 题校准 τ，160 题测试路由
5. **评估报告**：对比 P0 基线，分析元认知信号与路由效果

---

## 6. 关键脚本（待实现）

| 脚本 | 用途 |
|---|---|
| `scripts/medcog_signals.py` | 三维元认知评分 |
| `scripts/medcog_retriever.py` | BM25 KG 验证 + memory 检索（可被 import） |
| `scripts/run_medcog.py` | 五策略推理 + 门控路由 + 评估 |
| `docs/MedCoG复现结果.md` | 分析报告 |

---

## 7. 预期结论（诚实预设）

MedCoG 的元认知门控本质是**模型侧信号**调控知识使用。如果它在 MedXpertQA 上仍无法突破"检索增益为负"，将再次印证本研究的核心叙事——**知识侧信号与模型侧信号正交互补**才是稳健方向（而非单一模型侧元认知）；如果它有效，则说明 MedCoG 的 familiarity/knowledge density 信号比单纯 self_eval 更强，值得作为对比基线纳入论文。

无论哪种结果，都是一次有价值的对照实验。
