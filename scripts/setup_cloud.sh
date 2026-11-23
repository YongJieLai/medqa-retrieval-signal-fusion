#!/usr/bin/env bash
# 云服务器（RTX 4090 / 24GB）一键环境初始化 + 全流程执行
# 用法：bash scripts/setup_cloud.sh
set -euo pipefail

echo "=== [1/5] 安装依赖 ==="
# PyTorch 2.3.0 + cu121（匹配 AutoDL 4090 默认 CUDA 12.1 驱动）
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers faiss-gpu numpy openai sentencepiece rank-bm25 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple -q

echo "=== [2/5] 验证 GPU ==="
python3 -c "import torch; print('CUDA 可用:', torch.cuda.is_available(), '| 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== [3/5] 编码 PubMed 语料建 FAISS 索引（GPU） ==="
python3 scripts/build_pubmed_index.py

echo "=== [4/5] 验证索引 ==="
python3 -c "import faiss; ix=faiss.read_index('kg/medrag_pubmed/index.faiss'); print('索引条数:', ix.ntotal, '维度:', ix.d)"

echo "=== [5/5] 跑 8 配置推理（需 .env 里有有效 DASHSCOPE_API_KEY） ==="
python3 scripts/run_p0_baselines.py

echo ""
echo "=== 全部完成 ==="
echo "索引: kg/medrag_pubmed/index.faiss + corpus_meta.jsonl"
echo "结果: results/results_p0.jsonl"
