#!/bin/bash

# 强制 vLLM 的 worker 进程使用 'spawn' 启动方法。
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- 这里是关键修改 ---
# 指定使用物理ID为 2 和 3 的显卡。
export CUDA_VISIBLE_DEVICES=2

# 设置张量并行的大小，必须与上面指定的显卡数量一致。
TENSOR_PARALLEL_SIZE=1

# 显存使用率
GPU_MEM_UTILIZATION=0.5
CUDA_VISIBLE_DEVICES=1
echo "指定使用显卡 ID: ${CUDA_VISIBLE_DEVICES}"
echo "设置张量并行大小为: ${TENSOR_PARALLEL_SIZE}"
echo "尝试以 ${GPU_MEM_UTILIZATION} 的显存使用率启动 vLLM..."

# --model /home/workspace/lgq/shop/model/Qwen3-Embedding-8B \
# 启动 vLLM OpenAI API 服务器
python -m vllm.entrypoints.openai.api_server \
    --model /home/workspace/lgq/ms-swift/output/recall-qwen3-0.6-embedding/v0-20250815-045723/checkpoint-480 \
    --trust-remote-code \
    --port 8000 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTILIZATION}

echo "vLLM 服务已启动（或尝试启动失败）。"