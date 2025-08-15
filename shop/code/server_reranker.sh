#!/bin/bash

# 强制 vLLM 的 worker 进程使用 'spawn' 启动方法。
# 这是解决 "Cannot re-initialize CUDA in forked subprocess" 错误的关键。
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 设置一个合理的显存使用率上限 (例如 30%)
# 您可以根据 nvidia-smi 的输出动态调整这个值
GPU_MEM_UTILIZATION=0.8
CUDA_VISIBLE_DEVICES=3

echo "尝试以 ${GPU_MEM_UTILIZATION} 的显存使用率启动 vLLM..."
echo "指定使用显卡 ID: ${CUDA_VISIBLE_DEVICES}"
# 启动 vLLM OpenAI API 服务器
python -m vllm.entrypoints.openai.api_server \
    --model /home/workspace/lgq/shop/model/Qwen3-Reranker-8B \
    --trust-remote-code \
    --port 8001 \
    --gpu-memory-utilization ${GPU_MEM_UTILIZATION} \
    # 如果您希望使用多张显卡进行张量并行处理，请取消下面这行的注释，并设置相应的数量
    # 例如，使用2张显卡：
    # --tensor-parallel-size 2

echo "vLLM 服务已启动（或尝试启动失败）。"