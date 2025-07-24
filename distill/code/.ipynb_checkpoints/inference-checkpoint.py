# run_inference.py

import os
import json
import time
import requests
from tqdm import tqdm

# --- 1. API 和文件路径配置 ---
API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer xxx"
}
MODEL_NAME = "qwen2.5-72b-plan"

# 输入：包含 instruction 和 input 的测试集
INPUT_JSON_PATH = "/home/workspace/lgq/distill/data/20250714_sft_test_dataset.json" 
# 输出：包含模型推理结果的 JSON 文件
OUTPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250714/7b_inference_results.json"

def call_model_api(instruction, user_input):
    """调用模型 API 并返回模型的输出和延迟。"""
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "stop": ["<|im_end|>"]
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
        response.raise_for_status()
        response_json = response.json()
        model_output = response_json['choices'][0]['message']['content'].strip()
        delay = time.time() - start_time
        return model_output, delay
    except requests.exceptions.RequestException as e:
        # 使用 tqdm.write 确保打印信息不会扰乱进度条
        tqdm.write(f"\nAPI 请求失败: {e}")
        return f"ERROR: API_REQUEST_FAILED - {e}", -1
    except (KeyError, IndexError) as e:
        tqdm.write(f"\n解析 API 响应失败: {e}, 响应内容: {response.text}")
        return f"ERROR: API_RESPONSE_INVALID - {e}", -1

def main():
    """主推理流程"""
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
        print(f"成功加载测试数据集，共 {len(test_dataset)} 条。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 加载文件 '{INPUT_JSON_PATH}' 失败 - {e}")
        return

    inference_results = []
    
    print("开始进行模型推理...")
    
    # 遍历数据集，并使用 enumerate 来获取索引
    for i, item in enumerate(tqdm(test_dataset, desc="推理进度", ncols=100)):
        instruction = item.get("instruction", "")
        user_input = item.get("input", "")
        ground_truth = item.get("output", "")
        
        if not instruction or not user_input:
            tqdm.write(f"\n警告: 数据项缺少 'instruction' 或 'input'，已跳过。")
            continue
            
        model_output, delay = call_model_api(instruction, user_input)
        
        # *** 新增的实时打印部分 ***
        # 使用 tqdm.write 来避免与进度条冲突
        output_str = (
            f"\n"
            f"--- Sample {i+1}/{len(test_dataset)} ---\n"
            f" [用户输入]: {user_input}\n"
            f" [模型输出]: {model_output}\n"
            f" [标准答案]: {ground_truth}\n"
            f"----------------------"
        )
        tqdm.write(output_str)

        # 将原始数据和推理结果合并到一个新的字典中
        result_item = {
            "instruction": instruction,
            "input": user_input,
            "ground_truth": ground_truth,
            "model_output": model_output,
            "delay_seconds": delay
        }
        inference_results.append(result_item)
        
        # Optional: 短暂休眠，避免请求过于频繁
        # time.sleep(0.1)

    # 保存所有推理结果到一个文件
    with open(OUTPUT_INFERENCE_PATH, 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=4)

    print(f"\n推理完成！所有结果已保存到: {os.path.abspath(OUTPUT_INFERENCE_PATH)}")

if __name__ == "__main__":
    main()