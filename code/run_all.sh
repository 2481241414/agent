#!/bin/bash

# 1. 激活conda环境
echo "激活conda环境..."
# source ~/anaconda3/etc/profile.d/conda.sh
conda activate taxi

# 2. 切换工作目录
cd /home/workspace/LLaMA-Factory

# 3. 数据准备与格式转换
echo "开始数据转换..."

# 针对Markdown格式车列表数据
echo "处理Markdown格式数据..."
python /home/workspace/lgq/code/gen_csv_data_markdown.py
python /home/workspace/lgq/code/generate_sft_data_markdown.py

# # 针对JSON格式车列表数据
# echo "处理JSON格式数据..."
# python /home/workspace/lgq/code/gen_csv_data_json.py
# python /home/workspace/lgq/code/generate_sft_data_json.py

# # 合并所有SFT数据
# echo "合并SFT数据..."
# python /home/workspace/lgq/code/merge_data.py

# 4. 复制数据到训练目录
echo "准备训练数据..."
cp /home/workspace/lgq/code/merge_data.json /home/workspace/LLaMA-Factory/data/

# 若文件名不是 merge_data.json，请手动修改 dataset_info.json
echo "请核查 /home/workspace/LLaMA-Factory/data/dataset_info.json 是否已包含 merge_data.json 信息！"

# 5. 启动训练
echo "开始训练..."
bash /home/workspace/LLaMA-Factory/examples/train_lora/qwen2.5_72b_Instruct_lora_sft.sh

# 6. 合并lora到原模型
echo "合并lora参数到原模型..."
bash /home/workspace/LLaMA-Factory/examples/merge_lora/merge_lora_qwen2.5_72b_Instruct.sh

# 7. 启动部署
echo "启动VLLM部署..."
bash /home/workspace/LLaMA-Factory/examples/inference/bushu.sh

# 8. 新终端批量预测（可选）
echo "批量预测..."
python /home/workspace/lgq/code/inference3.py

# 9. 效果评估（可选）
echo "效果评估..."
python /home/workspace/lgq/code/analyse.py

echo "全部流程执行完毕！"
