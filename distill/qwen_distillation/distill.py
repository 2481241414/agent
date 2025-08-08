"""
DeepSpeed 单机多卡蒸馏示例（最终版）
教师：Qwen-72B（由 ZeRO-Inference 管理，避免 OOM）
学生：Qwen-7B（由 Trainer + DeepSpeed 管理，进行 LoRA 训练）
依赖：pip install deepspeed transformers datasets accelerate peft
启动：deepspeed --num_gpus=8 train_distill.py
"""

import json
import torch
import deepspeed
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# --- 配置部分 ---
MODEL_TEACHER = "/home/workspace/zm/hf_model/Qwen2.5-7B-Instruct"
MODEL_STUDENT = "/home/workspace/zm/hf_model/Qwen2.5-7B-Instruct"
DATA_PATH      = "/home/workspace/lgq/distill/data/20250714_sft_train_dataset.json"
DS_CONFIG_STUDENT = "/home/workspace/lgq/distill/qwen_distillation/ds_config.json" # 学生模型的 DeepSpeed 配置

# --- 数据加载与预处理 (保持不变) ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def preprocess(examples):
    prompts, responses = [], []
    for ex in examples:
        prompt = ex["instruction"].rstrip() + "\n\n【用户输入】" + ex["input"].rstrip()
        response = ex["output"]
        prompts.append(prompt)
        responses.append(response)
    return {"prompt": prompts, "response": responses}
dataset = Dataset.from_dict(preprocess(raw_data))

tokenizer = AutoTokenizer.from_pretrained(MODEL_STUDENT, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(ex):
    text = ex["prompt"] + ex["response"]
    model_inputs = tokenizer(text, truncation=True, max_length=1024, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs
tokenized_ds = dataset.map(tokenize_function, batched=False, remove_columns=dataset.column_names)

# --- 模型加载与配置 ---

# 1. 配置学生模型 (LoRA) - 这部分逻辑正确，保持不变
student_base = AutoModelForCausalLM.from_pretrained(
    MODEL_STUDENT,
    torch_dtype=torch.bfloat16,
)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
)
student = get_peft_model(student_base, peft_config)
print("--- 学生模型参数 ---")
student.print_trainable_parameters()


# 2. 配置教师模型 (ZeRO-Inference) - ✅ 这是核心修改区域
print("\n--- 配置教师模型 (ZeRO-Inference) ---")
# 先在 CPU 上加载模型结构，避免直接加载权重到 GPU 导致 OOM
teacher_base = AutoModelForCausalLM.from_pretrained(
    MODEL_TEACHER,
    torch_dtype=torch.bfloat16,
)

# 为教师模型定义一个独立的 DeepSpeed 配置，用于 ZeRO-Inference
# 这是方案一的核心
ds_config_teacher = {
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu", # 将参数卸载到 CPU，进一步节省显存
            "pin_memory": True
        }
    },
    "bf16": {"enabled": True},
    "train_batch_size": 16, # 这个值应等于 global_batch_size (per_device * grad_accum * num_gpus)
}

# 使用 deepspeed.initialize 来包装教师模型以进行推理
# 关键点：传入 model_parameters=[] 来告诉 DeepSpeed 不要为这个模型创建优化器
teacher_engine, *_ = deepspeed.initialize(
    model=teacher_base,
    config=ds_config_teacher,
    model_parameters=[], # 冻结教师模型
)
teacher_engine.eval()
print("教师模型已成功使用 ZeRO Stage 3 进行初始化，用于推理。")


# --- 自定义训练逻辑 ---
def distill_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    ce = torch.nn.functional.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1), ignore_index=-100
    )
    kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / T, dim=-1),
        torch.nn.functional.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean"
    ) * (T ** 2)
    return alpha * kl + (1 - alpha) * ce

class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        
        # 使用由 DeepSpeed 管理的 teacher_engine 进行前向传播
        # 不再需要 .to(device)，DeepSpeed 会自动处理跨卡通信
        with torch.no_grad():
            teacher_logits = teacher_engine(**inputs).logits
            
        student_logits = model(**inputs).logits
        loss = distill_loss(student_logits, teacher_logits, labels)
        return (loss, student_logits) if return_outputs else loss

# --- 训练参数与启动 ---
training_args = TrainingArguments(
    output_dir="/home/workspace/lgq/distill/qwen_distillation/distill_out_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    deepspeed=DS_CONFIG_STUDENT, # Trainer 使用为学生模型准备的配置文件
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    dataloader_drop_last=True,
    remove_unused_columns=False,
)

trainer = DistillTrainer(
    model=student,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()