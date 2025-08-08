import json
import requests
import pandas as pd
import re
import ast
import os

# 路径设置
# json_file = '/home/workspace/lgq/data/新标注1000条训练数据.json'
# output_file = '/home/workspace/lgq/data/inference/新标注1000条训练数据_ingerence_results_0611.csv'

#json_file = '/home/workspace/lgq/data/20250613/v1.0/merge_data_map.json'
#output_file = '/home/workspace/lgq/data/inference/merge_data_inference_results_0613.csv'

# json_file = '/home/workspace/lgq/data/20250613/v1.0/全新训练数据20250613_10percent_map.json'
# output_file = '/home/workspace/lgq/data/inference/全新训练数据20250613_10percent_inference_results_0613.csv'

# json_file = '/home/workspace/lgq/data/output/20250616/v3.0_FINAL/merge_data_map_v3.0_FINAL.json'
# output_file = '/home/workspace/lgq/data/inference/merge_data_map_v3.0_FINAL_inference_results_0616_test1111111.csv'

# json_file = '/home/workspace/lgq/data/output/20250619/v12.0_MultiTurnFix/test_dataset_05.json'
# output_file = '/home/workspace/lgq/data/inference/20250620/v2.0/test_dataset_05_inference_results_0619.csv'

# json_file = '/home/workspace/lgq/data/output/20250620/v16.0_Final/打车语料2.0(多轮query)_sft.json'
# output_file = '/home/workspace/lgq/data/inference/20250620/v1.0/打车语料2.0(多轮query)_sft_inference_results_0620.csv'

# json_file = '/home/workspace/lgq/data/output/20250625/v17.1_CorrectedGTSearch/test_dataset_05.json'
# output_file = '/home/workspace/lgq/data/inference/20250625/v1.0/test_dataset_05_inference_results_0625.csv'

# json_file = '/home/workspace/lgq/data/output/20250625/v4.2_FlexibleQueryCol/打车语料-2.0版-624-zhao - 车型理解_sft_output.json'
# output_file = '/home/workspace/lgq/data/inference/20250625/v2.0/打车语料-2.0版-624-zhao - 车型理解_inference_results_0625.csv'

# json_file = '/home/workspace/lgq/data/output/20250625/v18.0_GTFallbackFix/test_dataset_05.json'
# output_file = '/home/workspace/lgq/data/inference/20250702/v1.0/test_dataset_05_inference_results_0702.csv'

json_file = '/home/workspace/lgq/data/output/20250625/v4.2_FlexibleQueryCol/打车语料-2.0版-624-zhao - 车型理解_sft_output.json'
output_file = '/home/workspace/lgq/data/inference/20250702/v1.0/打车语料-2.0版-624-zhao - 车型理解_inference_results_0702.csv'

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
with open(json_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

url = "http://localhost:8000/v1/chat/completions"
# url = "http://aiplatac-pre-drcn.inner.cloud.hihonor.com/aip/access-agent/cloud-api/v1/task/take_taxi_sku_choose/aip_yoyo_assistant"
headers = {
    "Content-Type": "application/json",
    "Authorization": "xxx"
}

results = []

for i, item in enumerate(data_list):
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    # print(input_text)
    true_output = item.get('output', '')

    if not input_text:
        continue

    # 组装模型输入
    model_input = instruction + "\n" + input_text

    data = {
        "model": "Qwen25-72B-Instruct",
        "messages": [
            {"role": "user", "content": model_input}
        ],
        "max_tokens": 512000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        resp_json = response.json()
        model_output = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        
    except Exception as e:
        print(f"第{i+1}条发生错误: {e}")
        model_output = None

    # 提取query
    query = extract_query(input_text)
    # 提取可选车型组合
    option_combo = extract_option_combo(input_text)
    # 提取模型输出和真实输出的车型组合
    model_combo = extract_combo_from_output(model_output or '')
    true_combo = extract_combo_from_output(true_output)

    results.append({
        "query": query,
        "可选车型组合": option_combo,
        "模型输出车型组合": model_combo,
        "正确车型组合": true_combo
    })

    print(f"第{i+1}条完成")
    print(f"query: {query}")
    print(f"可选车型组合: {option_combo}")
    print(f"真实结果为：{true_output}，模型输出为：{model_output}")



# 保存为csv文件，包含4列
df = pd.DataFrame(
    results,
    columns=["query", "可选车型组合", "模型输出车型组合", "正确车型组合"]
)

# **自动创建输出文件夹（如果不存在）**
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"全部推理及处理完成，结果已保存到 {output_file}")
