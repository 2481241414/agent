import os
import re
import json
import ast
import pandas as pd
import time

def parse_vehicle_pairs_markdown(candidate_str):
    if not isinstance(candidate_str, str):
        return []
    lines = candidate_str.split('\n')
    pair_set = set()
    for line in lines:
        if not line or '车型大类' in line or '---' in line or '序号id' in line:
            continue
        fields = line.split('|')
        if len(fields) < 5:
            continue
        category = fields[3].strip()
        mode = fields[4].strip()
        if (not category or category == 'undefined') and (not mode or mode == 'undefined'):
            continue
        pair_set.add((category, mode))
    return [{category: mode} for category, mode in pair_set]

def parse_vehicle_pairs_json(candidate_str):
    if not isinstance(candidate_str, str):
        return []
    items = re.findall(r'\{.*?\}', candidate_str, re.DOTALL)
    pair_set = set()
    for item in items:
        try:
            obj = json.loads(item.replace("'", '"'))
            category = obj.get('category', '').strip()
            mode = obj.get('carModel', '').strip()
            if (not category or category == 'undefined') and (not mode or mode == 'undefined'):
                continue
            pair_set.add((category, mode))
        except Exception:
            continue
    return [{category: mode} for category, mode in pair_set]

def parse_vehicle_pairs_auto(candidate_str):
    if not isinstance(candidate_str, str):
        return []
    if '{' in candidate_str and ('category' in candidate_str and 'carModel' in candidate_str):
        return parse_vehicle_pairs_json(candidate_str)
    elif '|' in candidate_str:
        return parse_vehicle_pairs_markdown(candidate_str)
    else:
        return []

def get_app_from_candidates(candidate_str):
    if isinstance(candidate_str, str):
        if "高德地图" in candidate_str:
            return "高德地图"
        elif "滴滴" in candidate_str:
            return "滴滴"
    return "未知"

def get_final_value(row, key, answer_key):
    """
    优先用正确答案，否则用原有字段
    """
    val = row.get(key, '')
    if pd.notna(row.get(answer_key, None)) and str(row[answer_key]).strip() not in ['', 'nan']:
        val = row[answer_key]
    return val

def get_final_pairs(row, all_pairs):
    """
    优先用category正确答案和mode正确答案，不为空就用，否则用原始category/mode。
    只要category在车型组合key中出现，或者mode在value中出现，就召回该组合。
    新增逻辑：如果category或者mode为all，则输出[{all: all}]
    """
    def split_multi(s):
        if not s or str(s).lower() in ['nan', '无']:
            return []
        return [x.strip() for x in re.split(r'[，,、]', str(s)) if x.strip()]

    # 优先用正确答案，否则用原有字段
    cats = split_multi(get_final_value(row, 'category', 'category正确答案'))
    modes = split_multi(get_final_value(row, 'mode', 'mode正确答案'))

    # 新增：只要cats或modes有一个为all（不区分大小写），直接返回[{all: all}]
    if any(str(c).lower() == 'all' for c in cats) or any(str(m).lower() == 'all' for m in modes):
        return [{"all": "all"}]

    valid = []
    for pair in all_pairs:
        for k, v in pair.items():
            if (k in cats or v in modes) and pair not in valid:
                valid.append(pair)
    return valid

def build_instruction(app):
    # 只返回固定的说明，不包含具体的query和车型组合
    if app == "高德地图":
        instr = (
            "你是一个打车助手，你能根据用户的query去选择车型大类和车型小类的组合。\n"
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
            "确保最终的输出结果不重不漏，做到精确率和召回率均达到100%。\n"
        )
    else:
        instr = (
            "你是一个打车助手，你能根据用户的query去选择车型大类和车型小类的组合。\n"
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
            "确保最终的输出结果不重不漏，做到精确率和召回率均达到100%。\n"
        )
    return instr

def build_query(query, vehicle_pairs):
    # 变化内容：用户query和可选车型组合
    vehicle_pairs_str = json.dumps(vehicle_pairs, ensure_ascii=False)
    return f"用户的query是：{query}\n用户可以选择的车型组合为：{vehicle_pairs_str}"

def build_duolun_query(query, vehicle_pairs):
    # 变化内容：用户query和可选车型组合
    vehicle_pairs_str = json.dumps(vehicle_pairs, ensure_ascii=False)
    return f"用户的历史会话为：{history}，用户当前的query是：{query}\n用户可以选择的车型组合为：{vehicle_pairs_str}"

# --------------------------
# 新增：根据input字段去重
def dedup_by_input(sft_list):
    seen = set()
    result = []
    for item in sft_list:
        key = item['input']
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
# --------------------------

def process_csv_file(input_file, output_csv_file=None, output_json_file=None):
    df = pd.read_csv(input_file, encoding='utf-8', engine='python')
    # 车型组合列可能已存在，直接用
    if '车型组合' in df.columns:
        pair_col = df['车型组合'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()
    else:
        pair_col = []
        for idx, row in df.iterrows():
            candidate_str = row['车辆候选列表']
            vehicle_pairs = parse_vehicle_pairs_auto(candidate_str)
            pair_col.append(vehicle_pairs)
        df['车型组合'] = pair_col

    sft_list = []
    for idx, row in df.iterrows():
        vehicle_pairs = pair_col[idx]
        query = str(row.get('query', '')).strip()
        if not query:
            continue
        app = get_app_from_candidates(row.get('车辆候选列表', ''))
        instruction = build_instruction(app)
        input_query = build_query(query, vehicle_pairs)
        output_pairs = get_final_pairs(row, vehicle_pairs)
        output_str = f"{json.dumps(output_pairs, ensure_ascii=False)}"
        sft_list.append({
            "instruction": instruction,
            "input": input_query,
            "output": output_str
        })

    # **去重，保留input唯一**
    sft_list = dedup_by_input(sft_list)

    if output_csv_file:
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f'处理完成，结果已保存到：{output_csv_file}')

    if output_json_file:
        with open(output_json_file, 'w', encoding='utf-8') as fout:
            json.dump(sft_list, fout, ensure_ascii=False, indent=2)
        print(f"SFT样本已保存到：{output_json_file}")

def merge_json_files(file_list, output_file):
    merged_data = []
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f"警告：{file_name} 不是一个列表结构，已跳过。")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"合并完成，结果已保存到 {output_file}")

def batch_process(file_process_configs, merge_json_output=None):
    json_files = []
    for config in file_process_configs:
        print(f"正在处理：{config['input_csv']}")
        process_csv_file(
            config['input_csv'],
            config.get('output_csv'),
            config.get('output_json')
        )
        if config.get('output_json'):
            json_files.append(config['output_json'])
    if merge_json_output and json_files:
        merge_json_files(json_files, merge_json_output)

def get_today_str():
    return time.strftime("%Y%m%d", time.localtime())

VERSION = "v1.0"

def add_time_ver_dir(filepath):
    """
    在data目录下插入时间戳和版本号目录
    例如：
    /home/workspace/lgq/data/新标注1000条训练数据.json
    -> /home/workspace/lgq/data/20250609/v1.0/新标注1000条训练数据.json
    """
    parts = filepath.split(os.sep)
    # 找到data目录
    if "data" not in parts:
        raise ValueError("路径中未找到data目录，请检查路径格式！")
    data_idx = parts.index("data")
    # 新路径 = data之前 + data + 时间戳 + 版本号 + data之后
    new_parts = parts[:data_idx+1] + [get_today_str(), VERSION] + parts[data_idx+1:]
    new_path = os.sep.join(new_parts)
    # 创建目录
    target_dir = os.path.dirname(new_path)
    os.makedirs(target_dir, exist_ok=True)
    return new_path

if __name__ == "__main__":
    raw_file_process_configs = [
        # {
        #     'input_csv': '/home/workspace/lgq/data/新测试数据.csv',
        #     'output_csv': '/home/workspace/lgq/data/新测试数据_output.csv',
        #     'output_json': '/home/workspace/lgq/data/新测试数据_sft_output.json'
        # },
        {
            'input_csv': '/home/workspace/lgq/data/全新训练数据20250613_10percent.csv',
            'output_csv': '/home/workspace/lgq/data/全新训练数据20250613_10percent_output1.csv',
            'output_json': '/home/workspace/lgq/data/全新训练数据20250613_10percent_sft_output1.json'
        },
        # {
        #     'input_csv': '/home/workspace/lgq/data/全新训练数据20250613_90percent.csv',
        #     'output_csv': '/home/workspace/lgq/data/全新训练数据20250613_90percent_output.csv',
        #     'output_json': '/home/workspace/lgq/data/全新训练数据20250613_90percent_sft_output.json'
        # },
        # {
        #     'input_csv': '/home/workspace/lgq/data/表一.csv',
        #     'output_csv': '/home/workspace/lgq/data/表一output.csv',
        #     'output_json': '/home/workspace/lgq/data/markdown_sft_output.json'
        # },
        # {
        #     'input_csv': '/home/workspace/lgq/data/表二.csv',
        #     'output_csv': '/home/workspace/lgq/data/表二output.csv',
        #     'output_json': '/home/workspace/lgq/data/json_sft_output_2.json'
        # },
        # {
        #     'input_csv': '/home/workspace/lgq/data/表三.csv',
        #     'output_csv': '/home/workspace/lgq/data/表三output.csv',
        #     'output_json': '/home/workspace/lgq/data/json_sft_output_3.json'
        # },
        # {
        #     'input_csv': '/home/workspace/lgq/data/表四.csv',
        #     'output_csv': '/home/workspace/lgq/data/表四output.csv',
        #     'output_json': '/home/workspace/lgq/data/json_sft_output_4.json'
        # },
        # {
        #     'input_csv': '/home/workspace/lgq/data/表五.csv',
        #     'output_csv': '/home/workspace/lgq/data/表五output.csv',
        #     'output_json': '/home/workspace/lgq/data/json_sft_output_5.json'
        # },
        # 可以继续添加其它csv的配置
    ]
    file_process_configs = []
    for item in raw_file_process_configs:
        new_item = dict(item)
        if 'output_csv' in item and item['output_csv']:
            new_item['output_csv'] = add_time_ver_dir(item['output_csv'])
        if 'output_json' in item and item['output_json']:
            new_item['output_json'] = add_time_ver_dir(item['output_json'])
        file_process_configs.append(new_item)

    # merge_json_output = add_time_ver_dir('/home/workspace/lgq/data/新测试数据_map.json')
    merge_json_output = add_time_ver_dir('/home/workspace/lgq/data/全新训练数据20250613_map111.json')
    # merge_json_output = add_time_ver_dir('/home/workspace/lgq/data/merge_data_map.json')
    batch_process(file_process_configs, merge_json_output)
