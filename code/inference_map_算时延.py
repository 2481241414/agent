import json
import requests
import pandas as pd
import re
import ast
import time  # 1. 引入time模块

# 路径设置
# json_file = '/home/workspace/lgq/data/新标注1000条训练数据.json'
# output_file = '/home/workspace/lgq/data/inference/新标注1000条训练数据_ingerence_results_0611.csv'

#json_file = '/home/workspace/lgq/data/20250613/v1.0/merge_data_map.json'
#output_file = '/home/workspace/lgq/data/inference/merge_data_inference_results_0613.csv'

json_file = '/home/workspace/lgq/data/20250613/v1.0/全新训练数据20250613_10percent_map.json'
output_file = '/home/workspace/lgq/data/inference/全新训练数据20250613_10percent_inference_results_0613.csv'

def extract_query(text):
    """
    从 input 字段中提取“用户的query是：”后面的内容，直到“用户可以选择的车型组合为：”前
    """
    m = re.search(r'用户的query是：(.*?)用户可以选择的车型组合为：', text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # 兜底：如果没有“用户可以选择的车型组合为：”，则取“用户的query是：”后第一行
    if '用户的query是：' in text:
        return text.split('用户的query是：', 1)[1].split('\n')[0].strip()
    return text.strip()

def extract_option_combo(text):
    """
    从 input 字段中提取‘用户可以选择的车型组合为：’后面的车型组合列表（字符串转为列表）
    """
    m = re.search(r'用户可以选择的车型组合为：(.*)', text)
    if m:
        content = m.group(1).strip()
        try:
            # content 应该是 '[{...}, {...}, ...]'
            combo_list = ast.literal_eval(content)
            return str(combo_list)
        except Exception:
            # 尝试补全缺失的结尾括号
            try:
                combo_list = ast.literal_eval(content + ']')
                return str(combo_list)
            except Exception:
                return "[]"
    return "[]"

def extract_combo_from_output(text):
    """
    从 output 或模型输出中直接提取车型组合（JSON 列表字符串），返回标准字符串
    """
    try:
        combo_list = ast.literal_eval(text)
        return str(combo_list)
    except Exception:
        return "[]"

# 读取json文件
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
except FileNotFoundError:
    print(f"错误: 文件 '{json_file}' 未找到。请检查路径。")
    exit()

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "xxx"  # 请确保这是一个有效的Authorization Token
}

results = []
latencies = []  # 2. 用于存储每次推理的延迟

for i, item in enumerate(data_list):
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    true_output = item.get('output', '')
    
    latency = -1.0 # 初始化延迟为-1，表示失败或未执行
    model_output = None

    if not input_text:
        continue

    # 组装模型输入
    model_input = instruction + "\n" + input_text

    data = {
        "model": "Qwen2-72B-sft-lora",
        "messages": [
            {"role": "user", "content": model_input}
        ],
        "max_tokens": 512, # 建议设置一个合理的max_tokens，512000太大了
        "temperature": 0.7
    }

    try:
        # 3. 记录开始和结束时间
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        
        # 计算延迟
        latency = end_time - start_time
        latencies.append(latency)
        
        response.raise_for_status() # 检查HTTP请求是否成功
        resp_json = response.json()
        model_output = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        
    except requests.exceptions.RequestException as e:
        print(f"第{i+1}条发生网络请求错误: {e}")
        model_output = f"请求错误: {e}"
    except Exception as e:
        print(f"第{i+1}条发生未知错误: {e}")
        model_output = f"未知错误: {e}"

    # 提取query
    query = extract_query(input_text)
    # 提取可选车型组合
    option_combo = extract_option_combo(input_text)
    # 提取模型输出和真实输出的车型组合
    model_combo = extract_combo_from_output(model_output or '')
    true_combo = extract_combo_from_output(true_output)

    # 4. 将延迟也添加到结果中
    results.append({
        "query": query,
        "可选车型组合": option_combo,
        "模型输出车型组合": model_combo,
        "正确车型组合": true_combo,
        "inference_latency": latency # 添加延迟列
    })

    print(f"第{i+1}/{len(data_list)}条完成, 耗时: {latency:.4f}秒")
    # 为了简化控制台输出，可以注释掉下面的详细打印
    print(f"query: {query}")
    print(f"可选车型组合: {option_combo}")
    print(f"真实结果为：{true_output}，模型输出为：{model_output}")


# 5. 计算并打印平均延迟
if latencies:
    average_latency = sum(latencies) / len(latencies)
    print(f"\n全部推理完成。")
    print(f"成功请求了 {len(latencies)} 条数据。")
    print(f"平均推理延迟: {average_latency:.4f} 秒/条")
else:
    print("\n没有成功的推理请求，无法计算平均延迟。")


# 6. 保存为csv文件，包含新增的列
df = pd.DataFrame(
    results,
    columns=["query", "可选车型组合", "模型输出车型组合", "正确车型组合", "inference_latency"]
)

  
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"结果已保存到 {output_file}")