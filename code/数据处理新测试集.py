import os
import re
import json
import pandas as pd
import time
import ast # 引入ast库，用于安全地解析字符串格式的列表

# --- 1. Predefined Static Vehicle Lists and Maps ---
DIDI_VEHICLE_LIST = [
    {"index": "1", "category": "推荐", "carModel": "极速拼车", "price": 8},{"index": "2", "category": "推荐", "carModel": "特惠快车", "price": 12},{"index": "3", "category": "推荐", "carModel": "滴滴快车", "price": 13},{"index": "4", "category": "拼车", "carModel": "特价拼车", "price": 7},{"index": "5", "category": "拼车", "carModel": "极速拼车", "price": 8.5},{"index": "6", "category": "特价", "carModel": "花小猪外接版", "price": 9.8},{"index": "7", "category": "特价", "carModel": "特惠快车", "price": 8.8},{"index": "8", "category": "快速", "carModel": "花小猪正价版", "price": 10},{"index": "9", "category": "快速", "carModel": "曹操出行", "price": 11},{"index": "10", "category": "快速", "carModel": "特快出租车", "price": 10.3},{"index": "11", "category": "快速", "carModel": "滴滴特快", "price": 15.8},{"index": "12", "category": "快速", "carModel": "出租车", "price": 14},{"index": "13", "category": "快速", "carModel": "滴滴快车", "price": 16},{"index": "14", "category": "舒适", "carModel": "滴滴专车", "price": 42},{"index": "15", "category": "舒适", "carModel": "滴滴豪华车", "price": 68},{"index": "16", "category": "六座", "carModel": "六座专车", "price": 54},{"index": "17", "category": "更多", "carModel": "滴滴代驾", "price": 17},{"index": "18", "category": "更多", "carModel": "宠物出行", "price": 21},{"index": "19", "category": "更多", "carModel": "滴滴包车", "price": 39},{"index": "20", "category": "送货", "carModel": "汽车快送", "price": 25},{"index": "21", "category": "送货", "carModel": "跑腿", "price": 28},{"index": "22", "category": "特价", "carModel": "惊喜特价", "price": 9.9},{"index": "23", "category": "推荐", "carModel": "顺风车", "price": 22.5},{"index": "24", "category": "城际", "carModel": "城际拼车", "price": 56.5},{"index": "25", "category": "城际", "carModel": "顺风车", "price": 62},{"index": "26", "category": "推荐", "carModel": "花小猪外接版", "price": 15},{"index": "27", "category": "推荐", "carModel": "花小猪正价版", "price": 13.5},{"index": "28", "category": "推荐", "carModel": "滴滴专车", "price": 24},{"index": "29", "category": "推荐", "carModel": "惊喜特价", "price": 12.8}
]
GAODE_VEHICLE_LIST = [
    {"index": "1", "category": "拼车", "carModel": "极速拼车", "price": 7.2},{"index": "2", "category": "特惠快车", "carModel": "特惠快车", "price": 8.5},{"index": "3", "category": "经济型", "carModel": "曹操出行", "price": 7.9},{"index": "4", "category": "经济型", "carModel": "T3出行", "price": 8.3},{"index": "5", "category": "经济型", "carModel": "享道出行", "price": 9},{"index": "6", "category": "经济型", "carModel": "鞍马出行", "price": 10},{"index": "7", "category": "经济型", "carModel": "腾飞出行", "price": 10.5},{"index": "8", "category": "经济型", "carModel": "及时用车", "price": 9.5},{"index": "9", "category": "经济型", "carModel": "风韵出行", "price": 9.5},{"index": "10", "category": "经济型", "carModel": "环旅出行", "price": 8.9},{"index": "11", "category": "经济型", "carModel": "神州专车", "price": 9.5},{"index": "12", "category": "经济型", "carModel": "叮叮出行", "price": 10},{"index": "13", "category": "经济型", "carModel": "迪尔出行", "price": 10.5},{"index": "14", "category": "经济型", "carModel": "旅程约车", "price": 8},{"index": "15", "category": "经济型", "carModel": "聚优出租", "price": 11},{"index": "16", "category": "经济型", "carModel": "阳光出行", "price": 13},{"index": "17", "category": "经济型", "carModel": "首汽约车", "price": 11.5},{"index": "18", "category": "经济型", "carModel": "有序出行", "price": 12},{"index": "19", "category": "经济型", "carModel": "沛途出行", "price": 11.5},{"index": "20", "category": "经济型", "carModel": "有滴出行", "price": 12.5},{"index": "21", "category": "经济型", "carModel": "365约车", "price": 13},{"index": "22", "category": "经济型", "carModel": "南京出租", "price": 14},{"index": "23", "category": "经济型", "carModel": "腾飞新出租", "price": 12},{"index": "24", "category": "经济型", "carModel": "鞍马聚的新出租", "price": 16},{"index": "25", "category": "特快车", "carModel": "特快车", "price": 18.5},{"index": "26", "category": "出租", "carModel": "出租车", "price": 16},{"index": "27", "category": "优享型", "carModel": "曹操出行", "price": 17.5},{"index": "28", "category": "优享型", "carModel": "阳光出行", "price": 18},{"index": "29", "category": "优享型", "carModel": "T3出行", "price": 19},{"index": "30", "category": "优享型", "carModel": "首汽约车", "price": 19},{"index": "31", "category": "优享型", "carModel": "鞍马出行", "price": 19.5},{"index": "32", "category": "优享型", "carModel": "风韵出行", "price": 18.5},{"index": "33", "category": "优享型", "carModel": "腾飞出行", "price": 18},{"index": "34", "category": "优享型", "carModel": "有滴出行", "price": 18.5},{"index": "35", "category": "优享型", "carModel": "及时用车", "price": 19.5},{"index": "36", "category": "优享型", "carModel": "神州专车", "price": 19},{"index": "37", "category": "优享型", "carModel": "环旅出行", "price": 19.5},{"index": "38", "category": "优享型", "carModel": "叮叮出行", "price": 19.5},{"index": "39", "category": "优享型", "carModel": "沛途出行", "price": 20},{"index": "40", "category": "优享型", "carModel": "旅程约车", "price": 19},{"index": "41", "category": "专车", "carModel": "品质专车", "price": 32},{"index": "42", "category": "六座商务", "carModel": "六座商务", "price": 35},{"index": "43", "category": "送东西", "carModel": "高德秒送", "price": 26},{"index": "44", "category": "顺风车", "carModel": "顺风车", "price": 33},{"index": "45", "category": "特价拼车", "carModel": "特价拼车", "price": 12},{"index": "46", "category": "推荐", "carModel": "顺风车", "price": 17},{"index": "47", "category": "推荐", "carModel": "特价拼车", "price": 12},{"index": "48", "category": "经济", "carModel": "曹操出行", "price": 8.9},{"index": "49", "category": "经济", "carModel": "T3出行", "price": 8.5},{"index": "50", "category": "经济", "carModel": "享道出行", "price": 9},{"index": "51", "category": "经济", "carModel": "鞍马出行", "price": 7.8},{"index": "52", "category": "经济", "carModel": "腾飞出行", "price": 7.5},{"index": "53", "category": "经济", "carModel": "有序出行", "price": 8},{"index": "54", "category": "经济", "carModel": "神州专车", "price": 8},{"index": "55", "category": "经济", "carModel": "妥妥E行", "price": 7.9},{"index": "56", "category": "经济", "carModel": "及时用车", "price": 7.6},{"index": "57", "category": "经济", "carModel": "环旅出行", "price": 8},{"index": "58", "category": "经济", "carModel": "沛途出行", "price": 8.1},{"index": "59", "category": "经济", "carModel": "旅程约车", "price": 8.3},{"index": "60", "category": "经济", "carModel": "风韵出行", "price": 8.3},{"index": "61", "category": "经济", "carModel": "阳光出行", "price": 8.2},{"index": "62", "category": "经济", "carModel": "首汽约车", "price": 8},{"index": "63", "category": "经济", "carModel": "叮叮出行", "price": 7.6},{"index": "64", "category": "经济", "carModel": "迪尔出行", "price": 7.5},{"index": "65", "category": "经济", "carModel": "有滴出行", "price": 7},{"index": "66", "category": "经济", "carModel": "聚优出租", "price": 8},{"index": "67", "category": "经济", "carModel": "南京出租", "price": 7.4},{"index": "68", "category": "经济", "carModel": "腾飞新出租", "price": 7.9},{"index": "69", "category": "经济", "carModel": "鞍马聚的新出租", "price": 8.5},{"index": "70", "category": "优享型", "carModel": "享道出行", "price": 11},{"index": "71", "category": "推荐", "carModel": "品质专车", "price": 28},{"index": "72", "category": "经济", "carModel": "365约车", "price": 10},{"index": "73", "category": "优享型", "carModel": "旅程专车", "price": 29},{"index": "74", "category": "经济型", "carModel": "聚优出行", "price": 9},{"index": "75", "category": "出租车", "carModel": "鞍马聚的-高档出租车", "price": 15},{"index": "76", "category": "品质专车", "carModel": "风韵出行", "price": 35},{"index": "77", "category": "品质专车", "carModel": "鞍马出行", "price": 42},{"index": "78", "category": "品质专车", "carModel": "叮叮出行", "price": 46},{"index": "79", "category": "品质专车", "carModel": "首汽约车", "price": 45},{"index": "80", "category": "品质专车", "carModel": "腾飞出行", "price": 38},{"index": "81", "category": "品质专车", "carModel": "环旅出行", "price": 39},{"index": "82", "category": "品质专车", "carModel": "神州专车", "price": 45},{"index": "83", "category": "六座商务", "carModel": "享道出行", "price": 41},{"index": "84", "category": "六座商务", "carModel": "首汽约车", "price": 39},{"index": "85", "category": "出租车", "carModel": "鞍马聚的-低档", "price": 19},{"index": "86", "category": "六座商务", "carModel": "叮叮出行", "price": 44},{"index": "87", "category": "六座商务", "carModel": "腾飞出行", "price": 40},{"index": "88", "category": "六座商务", "carModel": "神州专车", "price": 50},{"index": "89", "category": "特惠快车", "carModel": "首汽约车", "price": 12},{"index": "90", "category": "特惠快车", "carModel": "T3出行", "price": 13},{"index": "91", "category": "特惠快车", "carModel": "享道出行", "price": 12.5},{"index": "92", "category": "特惠快车", "carModel": "曹操出行", "price": 13.8},{"index": "93", "category": "特惠快车", "carModel": "鞍马出行", "price": 13.4},{"index": "94", "category": "特惠快车", "carModel": "风韵出行", "price": 12.9},{"index": "95", "category": "特惠快车", "carModel": "有序出行", "price": 12.6},{"index": "96", "category": "特惠快车", "carModel": "腾飞出行", "price": 12.8},{"index": "97", "category": "特惠快车", "carModel": "阳光出行", "price": 13},{"index": "98", "category": "特惠快车", "carModel": "叮叮出行", "price": 12.2},{"index": "99", "category": "特惠快车", "carModel": "神州专车", "price": 12.5},{"index": "100", "category": "特惠快车", "carModel": "迪尔出行", "price": 12.6},{"index": "101", "category": "特惠快车", "carModel": "及时用车", "price": 12.5},{"index": "102", "category": "特惠快车", "carModel": "有滴出行", "price": 13.4},{"index": "103", "category": "特惠快车", "carModel": "环旅出行", "price": 13.2},{"index": "104", "category": "特惠快车", "carModel": "沛途出行", "price": 12},{"index": "105", "category": "特惠快车", "carModel": "聚优出行", "price": 14},{"index": "106", "category": "特惠快车", "carModel": "南京出租", "price": 16},{"index": "107", "category": "特惠快车", "carModel": "365约车", "price": 12.5},{"index": "108", "category": "特惠快车", "carModel": "腾飞新出租", "price": 12.3},{"index": "109", "category": "特惠快车", "carModel": "妥妥E行", "price": 13},{"index": "110", "category": "特惠快车", "carModel": "聚的新出租", "price": 11.8},{"index": "111", "category": "经济型", "carModel": "妥妥E行", "price": 14},{"index": "112", "category": "优享型", "carModel": "365约车", "price": 15},{"index": "113", "category": "品质专车", "carModel": "有序出行", "price": 48},{"index": "114", "category": "六座商务", "carModel": "特价专车", "price": 40},{"index": "115", "category": "特惠快车", "carModel": "旅程约车", "price": 16},{"index": "116", "category": "特惠快车", "carModel": "聚优出租", "price": 17},{"index": "117", "category": "特惠快车", "carModel": "聚的出租车", "price": 17.9},{"index": "118", "category": "品质专车", "carModel": "享道出行", "price": 38}
]

DIDI_VEHICLE_MAP = {item['index']: item for item in DIDI_VEHICLE_LIST}
GAODE_VEHICLE_MAP = {item['index']: item for item in GAODE_VEHICLE_LIST}

# --- 2. Helper Functions ---
def find_column_by_substring(df, substring):
    for col in df.columns:
        if substring in col: return col
    raise ValueError(f"无法在CSV中找到包含 '{substring}' 的列。可用的列: {list(df.columns)}")

def extract_query_from_input_col(text):
    if not isinstance(text, str) or not text.strip(): return ""
    try:
        data_list = json.loads(text)
        exts_str = data_list[0].get("exts")
        exts_json = json.loads(exts_str)
        query = exts_json.get("userHistoryInputContent", "")
        if query:
            cleaned_query = query.replace('\n', ' ').strip()
            return re.sub(r'\s+', ' ', cleaned_query)
        return ""
    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        return ""

# 使用“车列表”列来判断APP名称
def extract_app_from_context(text):
    if not isinstance(text, str): return "未知"
    if "高德" in text: return "高德地图"
    if "滴滴" in text: return "滴滴"
    return "未知"

def get_static_vehicle_list(app_name):
    if app_name == "滴滴": return DIDI_VEHICLE_LIST
    if app_name == "高德地图": return GAODE_VEHICLE_LIST
    return []

# 核心修正：使用ast从'模型2标准答案'列解析索引并查找
def extract_answers_by_index_lookup(row, ground_truth_col, app_name_col):
    app_name = row[app_name_col]
    output_text = row[ground_truth_col]
    
    if pd.isna(output_text) or not str(output_text).strip():
        return "无", "无"

    vehicle_map = DIDI_VEHICLE_MAP if app_name == "滴滴" else GAODE_VEHICLE_MAP
    
    indices = []
    try:
        parsed_obj = ast.literal_eval(str(output_text))
        if isinstance(parsed_obj, list):
             indices = parsed_obj
    except (ValueError, SyntaxError):
        return "无", "无"

    if not indices:
        return "[]", "[]"

    found_categories, found_modes = set(), set()
    for index_val in indices:
        vehicle = vehicle_map.get(str(index_val))
        if vehicle:
            found_categories.add(vehicle['category'])
            found_modes.add(vehicle['carModel'])
    
    if not found_categories and not found_modes:
        if indices == []:
            return "[]", "[]"
        return "无", "无"

    return "，".join(sorted(list(found_categories))), "，".join(sorted(list(found_modes)))

def get_final_value(row, key, answer_key):
    val = row.get(key, '')
    if pd.notna(row.get(answer_key, None)) and str(row[answer_key]).strip() not in ['', 'nan']:
        val = row[answer_key]
    return val

def get_final_pairs(row, all_pairs):
    def split_multi(s):
        if not s or str(s).lower() in ['nan', '无']: return []
        if str(s).strip() == '[]': return ["[]"]
        return [x.strip() for x in re.split(r'[，,、]', str(s)) if x.strip()]

    cats = split_multi(get_final_value(row, 'category', 'category正确答案'))
    modes = split_multi(get_final_value(row, 'mode', 'mode正确答案'))

    if cats == ["[]"] or modes == ["[]"]: return []
    if any(str(c).lower() == 'all' for c in cats) or any(str(m).lower() == 'all' for m in modes):
        return all_pairs
    if not cats and not modes: return []

    valid = []
    for pair in all_pairs:
        k, v = list(pair.items())[0]
        if (k in cats or v in modes) and pair not in valid:
            valid.append(pair)
    return valid


def build_instruction(app):
    if app == "高德地图":
        return ("你是一个高德打车助手，你能根据用户的query去选择车型大类和车型小类的组合。\n"
                "用户会给你一个query和可选车型组合。\n"
                "你要根据用户的query，只能从可选车型组合中选择合适的。\n"
                "输出格式如下：\n"
                "车型组合:[{大类1:小类1}, {大类2:小类2}]\n\n"
                "特殊情况说明：\n"
                "如果用户对车型没有明确要求，则车型组合字段为空。\n"
                "如果用户要求打所有车型，则输出所有可选的车型组合。\n"
                "当用户要求的车在可选列表中没有时，则输出车型组合:[]。\n"
                "注意：只能从可选择的车型组合中选择，不能自己造组合。\n"
                "忽略用户query中提到的价格，避免因为价格去选择错误的车型。\n"
                "确保最终的输出结果不重不漏，做到精确率和召回率均达到100%。\n")
    else: # 默认为滴滴
        return ("你是一个滴滴打车助手，你能根据用户的query去选择车型大类和车型小类的组合。\n"
                "用户会给你一个query和可选车型组合。\n"
                "你要根据用户的query，只能从可选车型组合中选择合适的。\n"
                "输出格式如下：\n"
                "[{大类1:小类1}, {大类2:小类2}]\n\n"
                "特殊情况说明：\n"
                "如果用户对车型没有明确要求，则车型组合字段为空。\n"
                "如果用户要求打所有车型，则输出所有可选的车型组合。\n"
                "当用户要求的车在可选列表中没有时，则输出车型组合:[]。\n"
                "注意：只能从可选择的车型组合中选择，不能自己造组合。\n"
                "忽略用户query中提到的价格，避免因为价格去选择错误的车型。\n"
                "确保最终的输出结果不重不漏，做到精确率和召回率均达到100%。\n")

def build_query(query, vehicle_pairs_for_prompt):
    vehicle_pairs_str = json.dumps(vehicle_pairs_for_prompt, ensure_ascii=False)
    return f"用户的query是：{query}\n用户可以选择的车型组合为：{vehicle_pairs_str}"

def dedup_by_input(sft_list):
    seen, result = set(), []
    for item in sft_list:
        if item['input'] not in seen:
            seen.add(item['input'])
            result.append(item)
    return result

# --- 3. Main Processing Logic ---
def process_csv_file(input_file, output_csv_file=None, output_json_file=None):
    df = pd.read_csv(input_file, encoding='utf-8', engine='python')

    # 明确指定列名，保证准确性
    query_source_col = '车型理解'
    app_context_col = '车列表'
    ground_truth_col = '模型2标准答案'

    # 检查核心列是否存在
    required_cols = [query_source_col, app_context_col, ground_truth_col]
    for col in required_cols:
        if col not in df.columns:
            # 兼容旧格式，如果'车型理解'不存在，则尝试从'input'列提取
            if col == '车型理解':
                try:
                    query_source_col = find_column_by_substring(df, 'input')
                    if query_source_col is None: raise ValueError
                    print(f"注意: 在文件 {input_file} 中未找到'车型理解'列, 将从 'input' 列提取query。")
                    df['query'] = df[query_source_col].apply(extract_query_from_input_col)
                except ValueError:
                    print(f"处理文件 {input_file} 失败: 既未找到 '车型理解' 也未找到 'input' 列。")
                    return
            else:
                 print(f"处理文件 {input_file} 失败: 缺少核心列 '{col}'。")
                 return
    
    # 如果query列不是从input中提取的，现在创建它
    if 'query' not in df.columns:
        df['query'] = df[query_source_col].astype(str).str.strip().replace('nan', '')

    # --- 数据预处理 ---
    # 2. 提取AppName
    df['appName'] = df[app_context_col].apply(extract_app_from_context)

    # 3. 生成固定的车列表
    df['车型组合_full'] = df['appName'].apply(get_static_vehicle_list)
    df['车型组合'] = df['车型组合_full'].apply(lambda l: sorted([{i['category']: i['carModel']} for i in l], key=lambda x: list(x.keys())[0]))
    df['车辆候选列表'] = df['车型组合_full'].apply(lambda x: json.dumps(x, ensure_ascii=False, indent=2))
    
    # 4. 根据索引查找，生成正确答案
    answers = df.apply(extract_answers_by_index_lookup, axis=1, args=(ground_truth_col, 'appName'))
    df['category正确答案'] = answers.apply(lambda x: x[0])
    df['mode正确答案'] = answers.apply(lambda x: x[1])

    # 5. 准备空列以便get_final_pairs函数使用
    if 'category' not in df.columns: df['category'] = ''
    if 'mode' not in df.columns: df['mode'] = ''
        
    # --- 生成SFT样本 ---
    sft_list = []
    for idx, row in df.iterrows():
        query = str(row.get('query', '')).strip()
        if not query: continue
        
        app = row['appName']
        vehicle_pairs_for_prompt = row['车型组合']
        
        instruction = build_instruction(app)
        input_query = build_query(query, vehicle_pairs_for_prompt)
        output_pairs = get_final_pairs(row, vehicle_pairs_for_prompt)
        
        output_str = f"{json.dumps(output_pairs, ensure_ascii=False)}" if app == "高德地图" else f"{json.dumps(output_pairs, ensure_ascii=False)}"
        sft_list.append({"instruction": instruction, "input": input_query, "output": output_str})

    sft_list = dedup_by_input(sft_list)
    
    # --- 输出文件 ---
    if output_csv_file:
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        desired_columns = [
            'query', '车辆候选列表', '车型组合', 'category', 'category正确答案', 'mode', 'mode正确答案'
        ]
        output_columns = [col for col in desired_columns if col in df.columns]
        df_output = df[output_columns]
        df_output.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f'处理完成，中间过程CSV已保存到：{output_csv_file}')

    if output_json_file:
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
        with open(output_json_file, 'w', encoding='utf-8') as fout:
            json.dump(sft_list, fout, ensure_ascii=False, indent=2)
        print(f"SFT样本已保存到：{output_json_file}")


# --- 4. Batch Processing and File Management ---
def merge_json_files(file_list, output_file):
    merged_data = []
    for file_name in file_list:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list): merged_data.extend(data)
                else: print(f"警告：{file_name} 不是一个列表结构，已跳过。")
        except FileNotFoundError:
            print(f"警告: 合并时未找到文件 {file_name}，可能该文件在处理时出错被跳过。")
    if merged_data:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"合并完成，结果已保存到 {output_file}")
    else:
        print("没有可合并的数据，未生成合并文件。")

def batch_process(file_process_configs, merge_json_output=None):
    json_files = []
    for config in file_process_configs:
        print(f"--- 正在处理：{config['input_csv']} ---")
        process_csv_file(config['input_csv'], config.get('output_csv'), config.get('output_json'))
        if config.get('output_json'): json_files.append(config['output_json'])
            
    if merge_json_output and json_files:
        print("--- 开始合并所有JSON文件 ---")
        merge_json_files(json_files, merge_json_output)

def get_today_str():
    return time.strftime("%Y%m%d", time.localtime())

VERSION = "v3.0_FINAL" # 标记为最终修正版

def add_time_ver_dir(filepath):
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    new_dir = os.path.join(base_dir, get_today_str(), VERSION)
    return os.path.join(new_dir, filename)

if __name__ == "__main__":
    base_input_path = '/home/workspace/lgq/data/'
    base_output_path = '/home/workspace/lgq/data/output'

    file_names = [
        '打车语料-2.0版 - 价格推理.csv',
        '打车语料-2.0版 - 车型理解.csv',
        '打车语料-2.0版 - appName场景.csv',
        '打车语料-2.0版 - 组合场景.csv'
    ]
    raw_file_process_configs = [
        {'input_csv': os.path.join(base_input_path, fname),
         'output_csv': os.path.join(base_output_path, fname.replace('.csv', '_output.csv')),
         'output_json': os.path.join(base_output_path, fname.replace('.csv', '_sft_output.json'))
        } for fname in file_names
    ]
    file_process_configs = []
    for item in raw_file_process_configs:
        new_item = dict(item)
        if 'output_csv' in item and item['output_csv']:
            new_item['output_csv'] = add_time_ver_dir(item['output_csv'])
        if 'output_json' in item and item['output_json']:
            new_item['output_json'] = add_time_ver_dir(item['output_json'])
        file_process_configs.append(new_item)

    merge_json_output = add_time_ver_dir(os.path.join(base_output_path, f'merge_data_map_{VERSION}.json'))
    
    batch_process(file_process_configs, merge_json_output=merge_json_output)