# 项目交接文档：医学问答检索分配研究

> 交接时间：2026-11-23
> 交接对象：接手本项目的 AI Agent
> 本文档自包含，读完即可独立继续工作，无需依赖任何对话历史。

---

## 0. 一句话任务

在专家级医学多选题 **MedXpertQA** 上，验证「知识侧 coverage × 模型侧 familiarity 信号正交互补」能否打通「检索增益为负」，并完成「推理增强方法评测」的 DeepSeek backbone 切换。

---

## 1. 项目背景与研究主线

### 1.1 核心研究问题

医学问答中，标准 RAG（检索增强生成）的增益普遍为负——「检索补知识」反而降低准确率。本项目系统排查根因，核心命题是：

> 用「知识侧信号 coverage（知识缺口估值）」与「模型侧信号 familiarity（模型熟悉度）」的**正交融合**，调控「该不该检索、用哪条证据」，从而打通检索增益为负。

### 1.2 评测数据集

- **MedXpertQA**：专家级 10 选 1 医学多选题，Text 子集共 2450 题（全带 label）
- 本阶段用 **200 题分层抽样**：Reasoning 152 / Understanding 48（`data/medxpertqa_p0_200.jsonl`）

---

## 2. 已完成的工作与结论（重要，别重复做）

### 2.1 已跑完的实验链

| 阶段 | 内容 | 结论 |
|---|---|---|
| P0 | 失效基线（8 配置，qwen-flash）| 检索增益全负，「检索补相关能提升准确率」被证伪 |
| P1 | M0 闭卷 + M1 标准 RAG（qwen-plus）| 闭卷 20.0%，标准 RAG 增益 −1%~−2.5% |
| P2 | M2 推理查询扩展 + M5 证据过滤 | M2 首次正增益（+3），但幅度小 |
| 检索升级 | BM25 → MedCPT 语义检索（173 万篇）| **悖论**：检索质量提升，准确率反降 |
| 门控实验 | familiarity/语义分数做「该不该检索」门控 | 单信号门控无效（corr≈0）|
| 正交性验证 | 真正 coverage × familiarity | 信号正交成立（+17 互补），门控撞天花板 |

### 2.2 三个决定性负面结果（核心结论）

1. **检索质量无关**：BM25→MedCPT 语义检索（相关性显著提升），M2 准确率反从 21.5% 降到 19.0%（−2.5%）。机制是「过度聚焦」——语义检索精准命中鉴别诊断干扰项，形成锚定偏差。

2. **单一信号无用**：familiarity、语义 top1、complexity、knowledge_density 与「检索增益」相关系数全部 |r|<0.05。

3. **信号正交但门控无效**：coverage × familiarity 确实正交（corr=+0.088），并集兜错题互补（48 题 > 单信号最大 31，互补 +17）。但融合门控准确率 = 单信号门控（22.60%），因 **M5 检索在 160 道闭卷错题里净救仅 3 题**——门控撞「检索救题能力极低」天花板。

### 2.3 关键数据速查

| 方法 | BM25 | MedCPT 语义 |
|---|---|---|
| M0 闭卷基线 | 20.0% | — |
| M1 k=3 | 19.0% | 17.5% |
| M2 推理查询扩展 | 21.5% | 19.0% |
| M5 证据过滤 | 20.5% | **21.5%** |

详细报告：`docs/MedXpertQA_RAG评测报告_检索质量悖论.md`（完整版）、`EXPERIMENT_SUMMARY.md`（精简版）、`docs/自包含研究报告_完整版.md`

---

## 3. 当前正在做的事（未完成，接手点）

### 3.1 背景：用户决定换 DeepSeek backbone

用户判断：**千问（qwen）的推理（CoT）能力可能较弱，而在这个数据集里 CoT 能力很关键**。因此切换到 DeepSeek 重新评测。

（注意：不是 qwen 额度用尽——用户明确说有额度，是主动换模型。）

### 3.2 当前卡点

已写好 `scripts/run_deepseek_cot_ablation.py`，测 4 个配置（200 题，温度 0）：

- deepseek-chat × Direct
- deepseek-chat × CoT
- deepseek-reasoner × Direct
- deepseek-reasoner × CoT

**已发现问题**：deepseek-reasoner 推理 token 长度极不稳定：
- 简单题：reasoning 458 token，3.9s 答完
- 难题（如 Text-817）：reasoning 消耗 8192 token 仍不够，55s 后 length 截断**拿不到答案**

本地跑过一轮但**卡住**（9 分钟连 10 题进度都没打印），已手动 kill。**该实验尚无结果。**

### 3.3 待解决的技术问题

1. **reasoner 推理长度失控**：部分难题 reasoning 超 8192 token，导致 content 为空、答案拿不到。需决策：
   - 加大 max_tokens（如 16384/32768），但成本高
   - 或用 `deepseek-chat` 为主（非推理模型，可控），reasoner 只作 M8 对照
2. **脚本无断点续传 + 结果不实时落盘**：`run_deepseek_cot_ablation.py` 是跑完全部 200 题才写文件，一旦中断全丢。**接手后必须改成每题实时写 + 断点续传**。

---

## 4. 下一步任务（明确指令）

### 4.1 必做：改脚本 + 重跑 DeepSeek CoT 对照

1. 修改 `scripts/run_deepseek_cot_ablation.py`：
   - 每题跑完立即 append 写入结果文件（实时落盘）
   - 启动时读取已有结果，跳过已完成题（断点续传）
   - reasoner 用足够大的 max_tokens（建议 16384），并对 finish_reason=length 的题标记「无答案」
2. 重跑 4 配置，输出 `results/deepseek_cot_ablation.jsonl`

### 4.2 分析目标（拿到结果后）

回答三个关键问题：
1. **deepseek-chat 的 CoT 是否终于为正**？（千问曾出现 CoT<Direct，即 10.5%<17.5%）
2. **deepseek-reasoner 是否整体强于 chat**？
3. **reasoner 的推理成本是否可控**？（有无大量题 length 截断拿不到答案）

如果 reasoner 大量题拿不到答案 + 成本爆炸，结论可能是：**不是要换更强推理模型，而是这个数据集的题本身超出当前模型推理能力边界**——这本身是可写的硬结论。

### 4.3 后续可选

若 DeepSeek 的 CoT 表现良好，则按《推理增强方法评测计划书》继续跑 M2（Self-Consistency）、M3（Medprompt）、M4（PCCR 推理预算）、M5（排除法）、M6（多智能体）等推理策略。计划书在资料库：`MedXpertQA_推理增强方法评测计划书.md`（nodeId `NY4CepihIpAMoy3v0bL82Z`）。

---

## 5. 环境与数据

### 5.1 代码仓库

- **GitHub**：`git@github.com:YongJieLai/medqa-retrieval-signal-fusion.git`（public）
- **分支**：main
- 已含全部脚本、文档、实验总结；`.env`、`data/`、`kg/`、`results/` 中间产物在 `.gitignore` 中**不入库**

### 5.2 关键数据（不在 git 里，需另行准备）

| 路径 | 大小 | 说明 |
|---|---|---|
| `data/medxpertqa_p0_200.jsonl` | 463K | 200 题分层抽样（**评测核心数据**）|
| `data/medxpertqa_text.jsonl` | 3.9M | Text 子集全量 2450 题 |
| `kg/primekg/` | 941M | PrimeKG 知识图谱（含 `entity_degree.json`）|
| `kg/medrag_pubmed/` | 4.2G | PubMed 语料 + BM25 索引 |

### 5.3 依赖

`requirements.txt`：numpy、pandas、networkx、openai、requests、scipy、matplotlib
（另需 `rank_bm25`、`faiss`、`transformers`、`torch`，未在 requirements 中，需手动装）

### 5.4 API Key（需手动配 .env，勿入库）

> ⚠️ 密钥不在本仓库中。真实密钥在项目负责人的本地文件 `.api_keys_local.md`（已被 gitignore）。接手前向项目负责人索取，或从原机器拷贝 `.env`。

- **DashScope**（qwen）：key 以 `sk-` 开头，base `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **DeepSeek**：key 以 `sk-` 开头，base `https://api.deepseek.com`（余额约 100 元）
- **GitHub token**：以 `ghp_` 开头

### 5.5 云服务器（可选，语义检索用）

- SSH：`ssh -p 51751 root@connect.bjb1.seetacloud.com`，密码 `DmhkCMs/CQ5NineYjCqzRsVgli+D+pwD`
- RTX 4090 / 24GB，存有 MedCPT 语义索引（173 万篇 FAISS，5.0G）
- 注意：可能已关机/过期

---

## 6. 关键约定与纪律

1. **防泄漏铁律**：检索 query 只用题干主体（去选项），轨迹库与评测集无交集。
2. **答案格式**：`Answer: <LETTER>` + `Reason: <一句话>`，解析用 `parse_answer()`（优先匹配 `Answer: X`，兜底从后往前找 A-J 字母）。
3. **信号必须答题前算**：coverage / familiarity 等信号不能用 label 泄漏。
4. **结果实时落盘**：任何长实验必须支持断点续传，避免中断全丢（这是上一轮的教训）。
5. **密钥不入库**：`.env`、`.api_keys_local.md` 已被 `.gitignore` 排除。

---

## 7. 给接手 Agent 的第一批动作清单

1. `git clone git@github.com:YongJieLai/medqa-retrieval-signal-fusion.git`
2. 配 `.env`（§5.4 的 key）
3. 装依赖（§5.3）
4. 准备数据 `data/medxpertqa_p0_200.jsonl`（如缺失，从原机器拷贝或重新抽样）
5. 改 `scripts/run_deepseek_cot_ablation.py`：实时落盘 + 断点续传 + reasoner 大 max_tokens
6. 重跑 DeepSeek CoT 对照，分析 §4.2 的三个问题
7. 产出结果 + 分析，更新 `EXPERIMENT_SUMMARY.md` 或写新报告
