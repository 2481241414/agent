# run_inference_async.py

import os
import json
import time
import asyncio
import aiohttp
from tqdm.asyncio import tqdm

# --- 1. API 和文件路径配置 ---
API_URL = "http://localhost:8000/v1/chat/completions"
# API_URL = "https://dashscope.aliyuncs.com/v1/chat/completions"
# API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer sk-4fcc85e2509649198bdcafa4e985ce6e"
}
# MODEL_NAME = "qwen2.5-72b-plan"
# MODEL_NAME = "qwen2.5-32b-instruct"
# MODEL_NAME = "qwen2.5-7b-instruct"
# MODEL_NAME = "qwen2.5-14b-instruct"
# MODEL_NAME = "qwen-plus"
MODEL_NAME = "qwen"
MAX_CONCURRENT_REQUESTS = 16

# INPUT_JSON_PATH = "/home/workspace/lgq/distill/data/20250714_sft_test_dataset.json"
# INPUT_JSON_PATH = "/home/workspace/lgq/distill/data/20250715_sft_formatted_dataset.json"
INPUT_JSON_PATH = "/home/workspace/lgq/distill/data/20250716_sft_formatted_dataset.json"
OUTPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250717/Qwen2.5-72b_results_async_2query_0717.json"

async def call_model_api_async(session, item, pbar):
    """
    使用 aiohttp 异步调用模型 API。
    """
    instruction = item.get("instruction", "")
    user_input = item.get("input", "")
    ground_truth = item.get("output", "")

    if not instruction or not user_input:
        return {
            "instruction": instruction,
            "input": user_input,
            "ground_truth": ground_truth,
            "model_output": "ERROR: MISSING_DATA",
            "delay_seconds": -1.0
        }

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

    start_time = time.time()
    result = {}
    try:
        async with session.post(API_URL, json=data, timeout=60) as response:
            response.raise_for_status()
            response_json = await response.json()
            model_output = response_json['choices'][0]['message']['content'].strip()
            delay = time.time() - start_time

            output_str = (
                f"\n--- Sample for input: '{user_input[:50]}...' ---\n"
                f" [模型输出]: {model_output}\n"
                f" [标准答案]: {ground_truth}\n"
                f"----------------------"
            )
            pbar.write(output_str)

            result = {
                "instruction": instruction,
                "input": user_input,
                "ground_truth": ground_truth,
                "model_output": model_output,
                "delay_seconds": delay
            }

    except Exception as e:
        pbar.write(f"\n请求失败: Input='{user_input[:50]}...', Error: {e}")
        result = {
            "instruction": instruction,
            "input": user_input,
            "ground_truth": ground_truth,
            "model_output": f"ERROR: {e}",
            "delay_seconds": -1.0
        }
    finally:
        pbar.update(1)
        return result

async def main():
    """主推理流程"""
    # <<< 修改点 1 >>>
    # 记录整个脚本的开始时间
    script_start_time = time.time()

    output_dir = os.path.dirname(OUTPUT_INFERENCE_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
        print(f"成功加载测试数据集，共 {len(test_dataset)} 条。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 加载文件 '{INPUT_JSON_PATH}' 失败 - {e}")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = []
        with tqdm(total=len(test_dataset), desc="并发推理进度", ncols=100) as pbar:
            for item in test_dataset:
                async def bound_task(item_to_process):
                    async with semaphore:
                        return await call_model_api_async(session, item_to_process, pbar)
                tasks.append(bound_task(item))
            
            inference_results = await asyncio.gather(*tasks)

    with open(OUTPUT_INFERENCE_PATH, 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=4)

    print(f"\n并发推理完成！所有结果已保存到: {os.path.abspath(OUTPUT_INFERENCE_PATH)}")

    # <<< 修改点 2 >>>
    # --- 在所有任务完成后，计算并显示统计数据 ---
    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time

    successful_requests = 0
    failed_requests = 0
    total_latency = 0.0

    for res in inference_results:
        # 只统计成功的请求
        if res["delay_seconds"] >= 0:
            successful_requests += 1
            total_latency += res["delay_seconds"]
        else:
            failed_requests += 1

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