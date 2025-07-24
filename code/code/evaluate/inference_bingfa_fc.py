# run_inference.py

import os
import json
import time
import asyncio
import aiohttp
import csv
from tqdm.asyncio import tqdm

# --- 1. API 和文件路径配置 ---
# API_URL = "http://localhost:8000/v1/chat/completions"
API_URL = "http://10.80.0.255:25114/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer xxx"
}
MODEL_NAME = "Qwen"
MAX_CONCURRENT_REQUESTS = 16

# --- 文件路径 ---
# INPUT_CSV_PATH = "/home/workspace/lgq/distill/data/指令集FC效果摸底数据集.ver1 - 250710（基于v1版修改pa2text后）.csv"
INPUT_CSV_PATH = r"D:\Agent\data\fc\FC能力验证数据集 - Sheet1.csv"
OUTPUT_INFERENCE_PATH = r"D:\Agent\data\20250722/fc/zhiyu_FC能力验证_inference_results.json"


async def call_model_api_async(session, item, pbar):
    """异步调用模型 API 并返回结果。"""
    user_prompt = item.get("user_prompt", "").strip().strip('"')
    ground_truth = item.get("DSL", "")
    # --- 新增: 提取 tag 列 ---
    tag = item.get("tag", "")

    # 基本的输入检查
    if not user_prompt or "用户的问题或任务是:" not in user_prompt:
        pbar.update(1)
        # --- 修改: 在返回结果中加入 tag ---
        return {"user_prompt": user_prompt, "ground_truth": ground_truth, "model_output": "ERROR: INVALID_USER_PROMPT_FORMAT", "delay_seconds": -1.0, "tag": tag}

    try:
        parts = user_prompt.split("用户的问题或任务是:")
        system_message = parts[0].strip()
        user_message = parts[1].strip().strip('"')
    except IndexError:
        pbar.update(1)
        # --- 修改: 在返回结果中加入 tag ---
        return {"user_prompt": user_prompt, "ground_truth": ground_truth, "model_output": "ERROR: FAILED_TO_PARSE_USER_PROMPT", "delay_seconds": -1.0, "tag": tag}

    # 构建API请求体
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 512, "temperature": 0.1, "stop": ["<|im_end|>"]
    }

    start_time = time.time()
    result = {}
    try:
        async with session.post(API_URL, headers=HEADERS, json=data, timeout=60) as response:
            response.raise_for_status()
            response_json = await response.json()
            model_output = response_json['choices'][0]['message']['content'].strip()
            delay = time.time() - start_time
            pbar.write(f"\n--- Input: '{user_message[:50]}...' ---\n [Model DSL]: {model_output}\n [GT DSL]:    {ground_truth}\n----------------------")
            # --- 修改: 在返回结果中加入 tag ---
            result = {"user_prompt": user_prompt, "ground_truth": ground_truth, "model_output": model_output, "delay_seconds": delay, "tag": tag}
    except Exception as e:
        pbar.write(f"\nRequest failed: Input='{user_message[:50]}...', Error: {e}")
        # --- 修改: 在返回结果中加入 tag ---
        result = {"user_prompt": user_prompt, "ground_truth": ground_truth, "model_output": f"ERROR: {e}", "delay_seconds": -1.0, "tag": tag}
    finally:
        pbar.update(1)
        return result

async def main():
    """主推理流程"""
    script_start_time = time.time()

    output_dir = os.path.dirname(OUTPUT_INFERENCE_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    try:
        with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_dataset = list(reader)
        print(f"成功加载 {len(test_dataset)} 条样本。")
    except Exception as e:
        print(f"错误: 加载文件 '{INPUT_CSV_PATH}' 失败 - {e}")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = []
        with tqdm(total=len(test_dataset), desc="并发推理", ncols=100) as pbar:
            for item in test_dataset:
                async def bound_task(item_to_process):
                    async with semaphore:
                        return await call_model_api_async(session, item_to_process, pbar)
                tasks.append(bound_task(item))
            inference_results = await asyncio.gather(*tasks)

    with open(OUTPUT_INFERENCE_PATH, 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=4)
    print(f"\n推理完成！所有结果已保存到: {os.path.abspath(OUTPUT_INFERENCE_PATH)}")

    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time
    successful_requests = sum(1 for res in inference_results if res["delay_seconds"] >= 0)
    failed_requests = len(inference_results) - successful_requests
    total_latency = sum(res["delay_seconds"] for res in inference_results if res["delay_seconds"] >= 0)
    average_latency = total_latency / successful_requests if successful_requests > 0 else 0
    qps = successful_requests / total_script_duration if total_script_duration > 0 else 0

    print("\n--- 推理性能统计 ---")
    print(f"  总样本数: {len(test_dataset)}")
    print(f"  成功请求: {successful_requests}")
    print(f"  失败请求: {failed_requests}")
    print(f"  脚本总耗时: {total_script_duration:.2f} 秒")
    print(f"  平均每条时延: {average_latency:.4f} 秒")
    print(f"  吞吐率 (QPS): {qps:.2f} 请求/秒")
    print("----------------------")

if __name__ == "__main__":
    asyncio.run(main())