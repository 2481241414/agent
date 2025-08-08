
---

# LLaMA-Factory 使用说明

---

## 📁 目录路径

- **LLaMA-Factory 地址：**  
  `/home/workspace/LLaMA-Factory`

- **运行训练脚本需要切换到该目录：**  
  ```bash
  cd /home/workspace/LLaMA-Factory
  ```

---

## 📥 数据准备

- **原始数据下载地址：**  
  [https://elink.e.hihonor.com/sheets/shtlrZBABaYcBxz76qKKPgFyYsc?sheet=7KShAF](https://elink.e.hihonor.com/sheets/shtlrZBABaYcBxz76qKKPgFyYsc?sheet=7KShAF)

- **注意事项：**  
  原始表格部分**列名占两行**，不便处理，已手动调整表头为一行，请注意核对。

---

## 🔄 训练数据转换脚本

### 1. 针对**Markdown格式**车列表数据

- **提取车大类、小类及真实答案到后四列：**
  ```bash
  python /home/workspace/lgq/code/gen_csv_data_markdown.py
  ```

- **转换为 SFT 所需 JSON 格式：**
  ```bash
  python /home/workspace/lgq/code/generate_sft_data_markdown.py
  ```

---

### 2. 针对**JSON格式**车列表数据

> JSON 数据有三个表，需**手动修改文件路径**分别运行

- **提取车大类、小类及真实答案到后四列：**
  ```bash
  python /home/workspace/lgq/code/gen_csv_data_json.py
  ```

- **转换为 SFT 所需 JSON 格式：**
  ```bash
  python /home/workspace/lgq/code/generate_sft_data_json.py
  ```

---

### 3. 合并所有 SFT 数据

- **合并数据：**
  ```bash
  python /home/workspace/lgq/code/merge_data.py
  ```

### 4. **数据复制粘贴说明**  
  将生成的数据**复制到** `/home/workspace/LLaMA-Factory/data`，请**先删除原有文件**后再粘贴。

- **注意：**  
  如果生成文件名不是 `merge_data.json`，请手动在  
  `/home/workspace/LLaMA-Factory/data/dataset_info.json`  
  **添加数据信息**。

---

## ⚙️ 训练与部署配置文件

- **训练配置文件：**  
  `/home/workspace/LLaMA-Factory/examples/train_lora/qwen2.5_72b_Instruct_lora_sft.yaml`

- **合并 lora 到原模型配置：**  
  `/home/workspace/LLaMA-Factory/examples/merge_lora/qwen2.5_72b_Instruct_lora_sft.yaml`

- **VLLM 部署配置文件（API 调用）：**  
  `/home/workspace/LLaMA-Factory/examples/inference/qwen2.5_72b_Instruct_sft.yaml`

---

## 🚀 启动脚本命令

- **训练：**
  ```bash
  bash /home/workspace/LLaMA-Factory/examples/train_lora/qwen2.5_72b_Instruct_lora_sft.sh
  ```

- **合并：**
  ```bash
  bash /home/workspace/LLaMA-Factory/examples/merge_lora/merge_lora_qwen2.5_72b_Instruct.sh
  ```

- **部署：**
  ```bash
  bash /home/workspace/LLaMA-Factory/examples/inference/bushu.sh
  ```

---

## 🧩 其他相关脚本

- **批量预测脚本：**
  ```bash
  python /home/workspace/lgq/code/inference3.py
  ```

- **效果评估脚本：**
  ```bash
  python /home/workspace/lgq/code/analyse.py
  ```

---

## 🔗 参考文档

- **LLaMA-Factory LoRA 微调参数说明：**  
  [https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html)

---

**如有问题请联系相关负责人。**