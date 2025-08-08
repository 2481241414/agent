#!/bin/bash

# 设置你想使用的GPU数量
num_gpus=8


deepspeed --num_gpus=${num_gpus} /home/workspace/lgq/distill/qwen_distillation/distill.py \
        --teacher_model "/home/workspace/LLaMA-Factory/lora_result/qwen2.5-7b-Instruct" \
        --student_model "/home/workspace/zm/hf_model/Qwen2.5-7B-Instruct" \
        --data_path "/home/workspace/lgq/distill/data/20250714_sft_train_dataset.json" \
        --deepspeed_config /home/workspace/lgq/distill/qwen_distillation/ds_config.json \
        --output_dir "/home/workspace/lgq/distill/qwen_distillation/distill_out/qwen-7b-distilled-output" \
        --epochs 3 \
        --learning_rate 5e-6