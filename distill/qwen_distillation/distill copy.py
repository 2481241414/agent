import os
import json
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from typing import Dict, Any, Tuple, List

# --- 1. 定义教师和学生模型 (保持不变) ---
teacher_model_name = "/home/workspace/LLaMA-Factory/output/qwen2.5-72b-Instruct_lora_sft_20250714_sft_train_dataset"
student_model_name = "/home/workspace/zm/hf_model/Qwen2.5-7B-Instruct"
output_dir = "/home/workspace/lgq/distill/qwen_distillation/distill_out/qwen1.5-7b-distilled"

# --- 2. 加载和预处理数据 (保持不变) ---
def create_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    formatted_texts = []
    for item in data:
        text = f"{item['instruction']}\n\n【用户输入】\n{item['input']}\n\n【输出】\n{item['output']}"
        formatted_texts.append(text)
    return Dataset.from_dict({"text": formatted_texts})

tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=4096)

full_dataset = create_dataset("/home/workspace/lgq/distill/data/20250714_sft_train_dataset.json")
tokenized_dataset = full_dataset.map(tokenize_function, batched=True)

# --- 3. 定义蒸馏 Trainer (核心修改部分) ---
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        # DeepSpeed 会自动处理模型到 GPU 的移动，我们无需手动 to(device)
        self.teacher_model.eval()

    def _get_all_losses(self, model, inputs) -> Dict[str, torch.Tensor]:
        """
        一个辅助函数，用于计算并返回所有相关的损失。
        """
        # 教师模型的输出 (在 no_grad 上下文中，不计算梯度)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # 学生模型的输出
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # 这是学生模型自己的标准损失 (例如交叉熵损失)
        student_loss = student_outputs.loss

        # 定义蒸馏参数
        alpha = 0.5  # 蒸馏损失和学生损失的权重
        temperature = 2.0  # 软化概率分布的温度

        # 计算蒸馏损失 (KL 散度)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
            log_target=False
        ) * (temperature ** 2)

        # 组合最终用于反向传播的总损失
        total_loss = alpha * student_loss + (1.0 - alpha) * distillation_loss

        return {
            "loss": total_loss,
            "student_loss": student_loss.detach(), # detach以防重复计算梯度
            "distill_loss": distillation_loss.detach()
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写 compute_loss。
        这个方法主要被 evaluation loop 调用。我们只返回总损失。
        """
        losses = self._get_all_losses(model, inputs)
        # The Trainer framework expects just the loss value from this method.
        return losses["loss"]

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        重写 training_step 以便记录所有损失。
        这是显示训练过程的关键！
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 计算所有损失
        all_losses = self._get_all_losses(model, inputs)
        total_loss = all_losses["loss"]

        # --- 日志记录 ---
        # 创建一个字典，包含我们想要显示的额外信息
        logs: Dict[str, float] = {
            "student_loss": all_losses["student_loss"].item(),
            "distill_loss": all_losses["distill_loss"].item(),
        }
        # 将这些日志添加到 Trainer 的状态中，它们会在 logging_step 时被打印出来
        self.state.log_history.append(logs)
        self.log(logs)
        
        # --- 反向传播 ---
        # 使用 DeepSpeed 或 Accelerate 处理梯度计算和反向传播
        if self.use_deepspeed:
            self.deepspeed.backward(total_loss)
        else:
            total_loss.backward()

        return total_loss.detach()


# --- 4. 设置训练 (基本保持不变) ---
def main():
    # 加载学生模型
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 加载教师模型
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10, # 每10步打印一次日志
        save_strategy="epoch",
        deepspeed="/home/workspace/lgq/distill/qwen_distillation/ds_config.json",
        bf16=True,
        # 添加 report_to="tensorboard" 或 "wandb" 可以获得更丰富的可视化界面
        # report_to="tensorboard", 
    )

    # 初始化我们修改后的 DistillationTrainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_model(output_dir)
    print(f"模型已保存到 {output_dir}")

if __name__ == "__main__":
    main()