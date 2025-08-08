import json
import requests
import pandas as pd
import re
import ast
import os

# --- 1. 静态数据定义 (从数据处理脚本复制过来) ---
# 将两个平台的完整车列表放在这里，以便进行反向查找
DIDI_VEHICLE_LIST = [
    {"index": "1", "category": "推荐", "carModel": "极速拼车", "price": 8},{"index": "2", "category": "推荐", "carModel": "特惠快车", "price": 12},{"index": "3", "category": "推荐", "carModel": "滴滴快车", "price": 13},{"index": "4", "category": "拼车", "carModel": "特价拼车", "price": 7},{"index": "5", "category": "拼车", "carModel": "极速拼车", "price": 8.5},{"index": "6", "category": "特价", "carModel": "花小猪外接版", "price": 9.8},{"index": "7", "category": "特价", "carModel": "特惠快车", "price": 8.8},{"index": "8", "category": "快速", "carModel": "花小猪正价版", "price": 10},{"index": "9", "category": "快速", "carModel": "曹操出行", "price": 11},{"index": "10", "category": "快速", "carModel": "特快出租车", "price": 10.3},{"index": "11", "category": "快速", "carModel": "滴滴特快", "price": 15.8},{"index": "12", "category": "快速", "carModel": "出租车", "price": 14},{"index": "13", "category": "快速", "carModel": "滴滴快车", "price": 16},{"index": "14", "category": "舒适", "carModel": "滴滴专车", "price": 42},{"index": "15", "category": "舒适", "carModel": "滴滴豪华车", "price": 68},{"index": "16", "category": "六座", "carModel": "六座专车", "price": 54},{"index": "17", "category": "更多", "carModel": "滴滴代驾", "price": 17},{"index": "18", "category": "更多", "carModel": "宠物出行", "price": 21},{"index": "19", "category": "更多", "carModel": "滴滴包车", "price": 39},{"index": "20", "category": "送货", "carModel": "汽车快送", "price": 25},{"index": "21", "category": "送货", "carModel": "跑腿", "price": 28},{"index": "22", "category": "特价", "carModel": "惊喜特价", "price": 9.9},{"index": "23", "category": "推荐", "carModel": "顺风车", "price": 22.5},{"index": "24", "category": "城际", "carModel": "城际拼车", "price": 56.5},{"index": "25", "category": "城际", "carModel": "顺风车", "price": 62},{"index": "26", "category": "推荐", "carModel": "花小猪外接版", "price": 15},{"index": "27", "category": "推荐", "carModel": "花小猪正价版", "price": 13.5},{"index": "28", "category": "推荐", "carModel": "滴滴专车", "price": 24},{"index": "29", "category": "推荐", "carModel": "惊喜特价", "price": 12.8}
]
GAODE_VEHICLE_LIST = [
    {"index": "1", "category": "拼车", "carModel": "极速拼车", "price": 7.2},{"index": "2", "category": "特惠快车", "carModel": "特惠快车", "price": 8.5},{"index": "3", "category": "经济型", "carModel": "曹操出行", "price": 7.9},{"index": "4", "category": "经济型", "carModel": "T3出行", "price": 8.3},{"index": "5", "category": "经济型", "carModel": "享道出行", "price": 9},{"index": "6", "category": "经济型", "carModel": "鞍马出行", "price": 10},{"index": "7", "category": "经济型", "carModel": "腾飞出行", "price": 10.5},{"index": "8", "category": "经济型", "carModel": "及时用车", "price": 9.5},{"index": "9", "category": "经济型", "carModel": "风韵出行", "price": 9.5},{"index": "10", "category": "经济型", "carModel": "环旅出行", "price": 8.9},{"index": "11", "category": "经济型", "carModel": "神州专车", "price": 9.5},{"index": "12", "category": "经济型", "carModel": "叮叮出行", "price": 10},{"index": "13", "category": "经济型", "carModel": "迪尔出行", "price": 10.5},{"index": "14", "category": "经济型", "carModel": "旅程约车", "price": 8},{"index": "15", "category": "经济型", "carModel": "聚优出租", "price": 11},{"index": "16", "category": "经济型", "carModel": "阳光出行", "price": 13},{"index": "17", "category": "经济型", "carModel": "首汽约车", "price": 11.5},{"index": "18", "category": "经济型", "carModel": "有序出行", "price": 12},{"index": "19", "category": "经济型", "carModel": "沛途出行", "price": 11.5},{"index": "20", "category": "经济型", "carModel": "有滴出行", "price": 12.5},{"index": "21", "category": "经济型", "carModel": "365约车", "price": 13},{"index": "22", "category": "经济型", "carModel": "南京出租", "price": 14},{"index": "23", "category": "经济型", "carModel": "腾飞新出租", "price": 12},{"index": "24", "category": "经济型", "carModel": "鞍马聚的新出租", "price": 16},{"index": "25", "category": "特快车", "carModel": "特快车", "price": 18.5},{"index": "26", "category": "出租", "carModel": "出租车", "price": 16},{"index": "27", "category": "优享型", "carModel": "曹操出行", "price": 17.5},{"index": "28", "category": "优享型", "carModel": "阳光出行", "price": 18},{"index": "29", "category": "优享型", "carModel": "T3出行", "price": 19},{"index": "30", "category": "优享型", "carModel": "首汽约车", "price": 19},{"index": "31", "category": "优享型", "carModel": "鞍马出行", "price": 19.5},{"index": "32", "category": "优享型", "carModel": "风韵出行", "price": 18.5},{"index": "33", "category": "优享型", "carModel": "腾飞出行", "price": 18},{"index": "34", "category": "优享型", "carModel": "有滴出行", "price": 18.5},{"index": "35", "category": "优享型", "carModel": "及时用车", "price": 19.5},{"index": "36", "category": "优享型", "carModel": "神州专车", "price": 19},{"index": "37", "category": "优享型", "carModel": "环旅出行", "price": 19.5},{"index": "38", "category": "优享型", "carModel": "叮叮出行", "price": 19.5},{"index": "39", "category": "优享型", "carModel": "沛途出行", "price": 20},{"index": "40", "category": "优享型", "carModel": "旅程约车", "price": 19},{"index": "41", "category": "专车", "carModel": "品质专车", "price": 32},{"index": "42", "category": "六座商务", "carModel": "六座商务", "price": 35},{"index": "43", "category": "送东西", "carModel": "高德秒送", "price": 26},{"index": "44", "category": "顺风车", "carModel": "顺风车", "price": 33},{"index": "45", "category": "特价拼车", "carModel": "特价拼车", "price": 12},{"index": "46", "category": "推荐", "carModel": "顺风车", "price": 17},{"index": "47", "category": "推荐", "carModel": "特价拼车", "price": 12},{"index": "48", "category": "经济", "carModel": "曹操出行", "price": 8.9},{"index": "49", "category": "经济", "carModel": "T3出行", "price": 8.5},{"index": "50", "category": "经济", "carModel": "享道出行", "price": 9},{"index": "51", "category": "经济", "carModel": "鞍马出行", "price": 7.8},{"index": "52", "category": "经济", "carModel": "腾飞出行", "price": 7.5},{"index": "53", "category": "经济", "carModel": "有序出行", "price": 8},{"index": "54", "category": "经济", "carModel": "神州专车", "price": 8},{"index": "55", "category": "经济", "carModel": "妥妥E行", "price": 7.9},{"index": "56", "category": "经济", "carModel": "及时用车", "price": 7.6},{"index": "57", "category": "经济", "carModel": "环旅出行", "price": 8},{"index": "58", "category": "经济", "carModel": "沛途出行", "price": 8.1},{"index": "59", "category": "经济", "carModel": "旅程约车", "price": 8.3},{"index": "60", "category": "经济", "carModel": "风韵出行", "price": 8.3},{"index": "61", "category": "经济", "carModel": "阳光出行", "price": 8.2},{"index": "62", "category": "经济", "carModel": "首汽约车", "price": 8},{"index": "63", "category": "经济", "carModel": "叮叮出行", "price": 7.6},{"index": "64", "category": "经济", "carModel": "迪尔出行", "price": 7.5},{"index": "65", "category": "经济", "carModel": "有滴出行", "price": 7},{"index": "66", "category": "经济", "carModel": "聚优出租", "price": 8},{"index": "67", "category": "经济", "carModel": "南京出租", "price": 7.4},{"index": "68", "category": "经济", "carModel": "腾飞新出租", "price": 7.9},{"index": "69", "category": "经济", "carModel": "鞍马聚的新出租", "price": 8.5},{"index": "70", "category": "优享型", "carModel": "享道出行", "price": 11},{"index": "71", "category": "推荐", "carModel": "品质专车", "price": 28},{"index": "72", "category": "经济", "carModel": "365约车", "price": 10},{"index": "73", "category": "优享型", "carModel": "旅程专车", "price": 29},{"index": "74", "category": "经济型", "carModel": "聚优出行", "price": 9},{"index": "75", "category": "出租车", "carModel": "鞍马聚的-高档出租车", "price": 15},{"index": "76", "category": "品质专车", "carModel": "风韵出行", "price": 35},{"index": "77", "category": "品质专车", "carModel": "鞍马出行", "price": 42},{"index": "78", "category": "品质专车", "carModel": "叮叮出行", "price": 46},{"index": "79", "category": "品质专车", "carModel": "首汽约车", "price": 45},{"index": "80", "category": "品质专车", "carModel": "腾飞出行", "price": 38},{"index": "81", "category": "品质专车", "carModel": "环旅出行", "price": 39},{"index": "82", "category": "品质专车", "carModel": "神州专车", "price": 45},{"index": "83", "category": "六座商务", "carModel": "享道出行", "price": 41},{"index": "84", "category": "六座商务", "carModel": "首汽约车", "price": 39},{"index": "85", "category": "出租车", "carModel": "鞍马聚的-低档", "price": 19},{"index": "86", "category": "六座商务", "carModel": "叮叮出行", "price": 44},{"index": "87", "category": "六座商务", "carModel": "腾飞出行", "price": 40},{"index": "88", "category": "六座商务", "carModel": "神州专车", "price": 50},{"index": "89", "category": "特惠快车", "carModel": "首汽约车", "price": 12},{"index": "90", "category": "特惠快车", "carModel": "T3出行", "price": 13},{"index": "91", "category": "特惠快车", "carModel": "享道出行", "price": 12.5},{"index": "92", "category": "特惠快车", "carModel": "曹操出行", "price": 13.8},{"index": "93", "category": "特惠快车", "carModel": "鞍马出行", "price": 13.4},{"index": "94", "category": "特惠快车", "carModel": "风韵出行", "price": 12.9},{"index": "95", "category": "特惠快车", "carModel": "有序出行", "price": 12.6},{"index": "96", "category": "特惠快车", "carModel": "腾飞出行", "price": 12.8},{"index": "97", "category": "特惠快车", "carModel": "阳光出行", "price": 13},{"index": "98", "category": "特惠快车", "carModel": "叮叮出行", "price": 12.2},{"index": "99", "category": "特惠快车", "carModel": "神州专车", "price": 12.5},{"index": "100", "category": "特惠快车", "carModel": "迪尔出行", "price": 12.6},{"index": "101", "category": "特惠快车", "carModel": "及时用车", "price": 12.5},{"index": "102", "category": "特惠快车", "carModel": "有滴出行", "price": 13.4},{"index": "103", "category": "特惠快车", "carModel": "环旅出行", "price": 13.2},{"index": "104", "category": "特惠快车", "carModel": "沛途出行", "price": 12},{"index": "105", "category": "特惠快车", "carModel": "聚优出行", "price": 14},{"index": "106", "category": "特惠快车", "carModel": "南京出租", "price": 16},{"index": "107", "category": "特惠快车", "carModel": "365约车", "price": 12.5},{"index": "108", "category": "特惠快车", "carModel": "腾飞新出租", "price": 12.3},{"index": "109", "category": "特惠快车", "carModel": "妥妥E行", "price": 13},{"index": "110", "category": "特惠快车", "carModel": "聚的新出租", "price": 11.8},{"index": "111", "category": "经济型", "carModel": "妥妥E行", "price": 14},{"index": "112", "category": "优享型", "carModel": "365约车", "price": 15},{"index": "113", "category": "品质专车", "carModel": "有序出行", "price": 48},{"index": "114", "category": "六座商务", "carModel": "特价专车", "price": 40},{"index": "115", "category": "特惠快车", "carModel": "旅程约车", "price": 16},{"index": "116", "category": "特惠快车", "carModel": "聚优出租", "price": 17},{"index": "117", "category": "特惠快车", "carModel": "聚的出租车", "price": 17.9},{"index": "118", "category": "品质专车", "carModel": "享道出行", "price": 38}
]


# --- 2. 辅助函数 ---
def create_reverse_map(vehicle_list):
    """根据车列表，创建一个 (category, carModel) -> index 的反向映射。"""
    reverse_map = {}
    for item in vehicle_list:
        # 使用元组作为键，因为字典不能作为键
        key = (item['category'], item['carModel'])
        # 处理一个 (category, carModel) 对应多个 index 的情况，虽然目前数据没有，但这是良好实践
        if key not in reverse_map:
            reverse_map[key] = []
        reverse_map[key].append(item['index'])
    return reverse_map

# 创建全局的反向映射表
DIDI_REVERSE_MAP = create_reverse_map(DIDI_VEHICLE_LIST)
GAODE_REVERSE_MAP = create_reverse_map(GAODE_VEHICLE_LIST)

def get_app_name_from_options(option_str):
    """根据可选车型组合的字符串内容，判断是滴滴还是高德。"""
    # 使用一些独特的车型来判断
    if "'T3出行'" in option_str or "'高德秒送'" in option_str:
        return "gaode"
    if "'滴滴快车'" in option_str or "'滴滴代驾'" in option_str:
        return "didi"
    # 默认或无法判断时返回滴滴（或根据你的数据分布选择默认值）
    return "didi"

def extract_query(text):
    m = re.search(r'用户的query是：(.*?)用户可以选择的车型组合为：', text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    if '用户的query是：' in text:
        return text.split('用户的query是：', 1)[1].split('\n')[0].strip()
    return text.strip()

def extract_option_combo(text):
    m = re.search(r'用户可以选择的车型组合为：(.*)', text)
    if m:
        return m.group(1).strip()
    return "[]"

def convert_to_indexed_format(combo_str, reverse_map):
    """将模型输出的车型组合列表，转换为带索引的格式。"""
    if not combo_str:
        return "[]"
    
    try:
        combo_list = ast.literal_eval(combo_str)
        if not isinstance(combo_list, list):
            return "[]"
    except (ValueError, SyntaxError):
        return "[]"

    indexed_list = []
    for pair_dict in combo_list:
        if not isinstance(pair_dict, dict) or not pair_dict:
            continue
        
        category, car_model = list(pair_dict.items())[0]
        key = (category, car_model)
        
        # 从反向映射中查找索引
        indices = reverse_map.get(key)
        if indices:
            # 一个(category, carModel)可能对应多个index，全部加入
            for index in indices:
                indexed_list.append({index: pair_dict})

    # 为了保持一致性，可以对结果按 index 排序
    indexed_list.sort(key=lambda x: int(list(x.keys())[0]))
    
    return str(indexed_list) # 以字符串形式返回

# --- 3. 主逻辑 ---

# 路径设置 (请根据你的实际情况修改)
json_file = '/home/workspace/lgq/data/output/20250625/v4.2_FlexibleQueryCol/打车语料-2.0版-624-zhao - 车型理解_sft_output.json'
output_file = '/home/workspace/lgq/data/inference/20250625/v2.0/打车语料-2.0版-624-zhao - 车型理解_inference_results_0625_indexed.csv'


# 读取json文件
with open(json_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# API URL 和 Headers (请根据你的实际情况修改)
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "xxx"
}

results = []

for i, item in enumerate(data_list):
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    true_output_str = item.get('output', '')

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
        model_output_str = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
    except Exception as e:
        print(f"第{i+1}条发生错误: {e}")
        model_output_str = '' # 出错时给一个空字符串

    # 提取 query 和可选车型组合
    query = extract_query(input_text)
    option_combo_str = extract_option_combo(input_text)

    # 判断APP类型并选择正确的反向映射表
    app_name = get_app_name_from_options(option_combo_str)
    reverse_map_to_use = DIDI_REVERSE_MAP if app_name == "didi" else GAODE_REVERSE_MAP
    
    # 转换模型输出和正确答案为带索引的格式
    model_combo_indexed = convert_to_indexed_format(model_output_str, reverse_map_to_use)
    true_combo_indexed = convert_to_indexed_format(true_output_str, reverse_map_to_use)

    results.append({
        "query": query,
        "可选车型组合": option_combo_str,
        "模型输出车型组合(带索引)": model_combo_indexed,
        "正确车型组合(带索引)": true_combo_indexed
    })

    print(f"第{i+1}条完成")
    print(f"query: {query}")
    print(f"真实结果(带索引)为：{true_combo_indexed}，模型输出(带索引)为：{model_combo_indexed}")


# 保存为csv文件
df = pd.DataFrame(
    results,
    columns=["query", "可选车型组合", "模型输出车型组合(带索引)", "正确车型组合(带索引)"]
)

# 自动创建输出文件夹
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"全部推理及处理完成，结果已保存到 {output_file}")