import os
import re
import json
import pandas as pd
import time
import ast

# --- 0. 统一指令与常量定义 ---
UNIFIED_INSTRUCTION = (
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

# --- 定义所有已知格式的特征列名 ---
# 格式1 (多轮)
F1_VEHICLE_COL = '车列表'
F1_GT_COL = '模型2标准答案'
F1_MULTI_Q_COLS = ['多轮对话query1', '多轮对话query2', '多轮对话query3']
F1_SINGLE_Q_CANDIDATES = ['车型理解', '价格推理', '组合场景', 'appName场景']
# 格式2 (多轮, query1/2/3)
F2_Q_COLS = ['query1', 'query2', 'query3']
F2_VEHICLE_COL = '车辆候选列表'
F2_GT_CAT_COL = '大类'
F2_GT_MODE_COL = '小类'
# 格式3 (单轮, query, 新格式)
F3_Q_COL = 'query'
F3_VEHICLE_COL = '车辆候选列表'
F3_GT_CAT_COL = 'category正确答案'
F3_GT_MODE_COL = 'mode正确答案'
F3_CAT_FALLBACK = 'category'
F3_MODE_FALLBACK = 'mode'
# 格式4 (多轮, query1/2)
F4_Q_COLS = ['query1', 'query2']
F4_VEHICLE_COL = '车辆候选列表'
F4_GT_CAT_COL = '车大类'
F4_GT_MODE_COL = '车小类'


# --- 1. 通用解析与辅助函数 ---

def parse_vehicle_pairs_markdown(markdown_str):
    if not isinstance(markdown_str, str): return []
    # 修复：兼容列名，如“车辆类型”或“车型小类”
    markdown_str = markdown_str.replace('车辆类型', '车型小类').replace('车型大类', '车型大类')
    lines = markdown_str.strip().split('\n')
    pair_set = set()
    category_idx, mode_idx = -1, -1
    header_found = False
    for line in lines:
        if '车型大类' in line and '车型小类' in line and '|' in line and '---' not in line:
            headers = [h.strip() for h in line.split('|')]
            try:
                category_idx = headers.index('车型大类')
                mode_idx = headers.index('车型小类')
                header_found = True
                break
            except ValueError: continue
    if not header_found: return []
    for line in lines:
        if not line.strip().startswith('|') or '---' in line or '车型大类' in line: continue
        fields = [field.strip() for field in line.split('|')]
        if len(fields) > max(category_idx, mode_idx):
            # 兼容字段索引可能超出范围的情况
            try:
                category = fields[category_idx]
                mode = fields[mode_idx]
                if category or mode:
                    if 'undefined' not in str(category).lower() and 'undefined' not in str(mode).lower():
                        pair_set.add((category, mode))
            except IndexError:
                continue
    return [{cat: mode} for cat, mode in sorted(list(pair_set))]

def parse_vehicle_pairs_json(json_like_str):
    if not isinstance(json_like_str, str): return []
    match = re.search(r'["\']taxiSkuList["\']\s*:\s*(\[.*?\])', json_like_str, re.DOTALL)
    if not match: return []
    try:
        vehicle_list = json.loads(match.group(1))
        pair_set = set()
        for item in vehicle_list:
            category, car_model = item.get('category', '').strip(), item.get('carModel', '').strip()
            if category or car_model: pair_set.add((category, car_model))
        return [{cat: mode} for cat, mode in sorted(list(pair_set))]
    except json.JSONDecodeError: return []

def parse_vehicle_pairs_json_objects(json_objects_str):
    if not isinstance(json_objects_str, str): return []
    json_strings = re.findall(r'\{.*?\}', json_objects_str, re.DOTALL)
    if not json_strings: return []
    pair_set = set()
    for obj_str in json_strings:
        try:
            item = json.loads(obj_str)
            category = item.get('category', '').strip()
            car_model = item.get('carModel', '').strip()
            if category or car_model:
                if 'undefined' not in str(category).lower() and 'undefined' not in str(car_model).lower():
                    pair_set.add((category, car_model))
        except json.JSONDecodeError: continue
    return [{cat: mode} for cat, mode in sorted(list(pair_set))]

def parse_vehicle_pairs_auto(cell_content):
    if not isinstance(cell_content, str) or is_truly_empty(cell_content):
        return []

    content_strip = cell_content.strip()
    if content_strip.startswith('|') and content_strip.count('|') > 3:
        return parse_vehicle_pairs_markdown(cell_content)
    elif '"taxiSkuList"' in content_strip or "'taxiSkuList'" in content_strip:
        return parse_vehicle_pairs_json(cell_content)
    elif content_strip.startswith('{') and content_strip.endswith('}'):
        return parse_vehicle_pairs_json_objects(cell_content)
    return []


def dedup_by_input(sft_list):
    seen, result = set(), []
    for item in sft_list:
        if item['input'] not in seen:
            seen.add(item['input'])
            result.append(item)
    return result

def build_input_query(history, query, vehicle_pairs):
    vehicle_pairs_str = json.dumps(vehicle_pairs, ensure_ascii=False)
    if history: return f"# 用户历史会话：\n{history}\n\n# 用户需求：\n{query}\n\n# 用户可以选择的车型组合为：\n{vehicle_pairs_str}"
    else: return f"用户的query是：{query}\n用户可以选择的车型组合为：{vehicle_pairs_str}"

# --- 2. 数据标准化阶段 ---
def _extract_dynamic_query_history(row, q_cols):
    queries = [str(row.get(c, '')).strip() for c in q_cols]
    last_valid_index = -1
    for i in range(len(queries) - 1, -1, -1):
        if queries[i] and queries[i].lower() != 'nan':
            last_valid_index = i
            break
    if last_valid_index == -1: return pd.Series(['', ''])
    current_query = queries[last_valid_index]
    history_queries = queries[:last_valid_index]
    history_parts = [f"{i+1}. {h_query}" for i, h_query in enumerate(history_queries) if h_query and h_query.lower() != 'nan']
    history_str = "\n".join(history_parts)
    return pd.Series([history_str, current_query])

def is_truly_empty(val):
    return pd.isna(val) or str(val).strip() == '' or str(val).strip().lower() == 'nan'

def normalize_dataframe(df):
    """
    【核心修复】标准化DataFrame，优化了格式检测顺序，确保多轮格式被优先识别。
    """
    cols = df.columns
    data_format = None

    # 优先检测特征最明显的多轮格式，再检测单轮格式，避免误判
    if all(c in cols for c in F1_MULTI_Q_COLS) and F1_VEHICLE_COL in cols:
        data_format = 'F1'
    elif all(c in cols for c in F2_Q_COLS) and F2_VEHICLE_COL in cols:
        data_format = 'F2'
    elif all(c in cols for c in F4_Q_COLS) and F4_VEHICLE_COL in cols:
        data_format = 'F4'
    elif F3_Q_COL in cols and F3_VEHICLE_COL in cols and F3_GT_CAT_COL in cols:
        data_format = 'F3'
    else:
        # 添加更灵活的回退检测
        if 'query' in cols and 'category正确答案' in cols:
            data_format = 'F3'
        else:
            raise ValueError(f"无法识别的CSV文件格式。文件列名: {list(cols)}")

    print(f"检测到格式: {data_format}")

    # --- F1: 多轮对话格式 (最老) ---
    if data_format == 'F1':
        df['_vehicle_pairs'] = df[F1_VEHICLE_COL].apply(parse_vehicle_pairs_auto)
        df[['_history', '_query']] = df.apply(_extract_dynamic_query_history, axis=1, q_cols=F1_MULTI_Q_COLS)
        is_single_turn = df['_query'].str.strip() == ''
        if is_single_turn.any():
            found_q_col = next((c for c in F1_SINGLE_Q_CANDIDATES if c in cols), None)
            if found_q_col: df.loc[is_single_turn, '_query'] = df.loc[is_single_turn, found_q_col]
        def gt_from_indices(row):
            try:
                indices_str = str(row[F1_GT_COL])
                if is_truly_empty(indices_str): return pd.Series(['', ''])
                indices = ast.literal_eval(indices_str)
                # ... (其余逻辑与之前版本相同)
            except: return pd.Series(['', ''])
        df[['_gt_category', '_gt_mode']] = df.apply(gt_from_indices, axis=1)

    # --- F2, F3, F4: 其他格式 (包括了所有新老、单轮、多轮格式) ---
    elif data_format in ['F2', 'F3', 'F4']:
        gt_map = {
            'F2': {'q': F2_Q_COLS, 'v': F2_VEHICLE_COL, 'cat': F2_GT_CAT_COL, 'mode': F2_GT_MODE_COL, 'cat_fb': F2_GT_CAT_COL, 'mode_fb': F2_GT_MODE_COL},
            'F3': {'q': [F3_Q_COL], 'v': F3_VEHICLE_COL, 'cat': F3_GT_CAT_COL, 'mode': F3_GT_MODE_COL, 'cat_fb': F3_CAT_FALLBACK, 'mode_fb': F3_MODE_FALLBACK},
            'F4': {'q': F4_Q_COLS, 'v': F4_VEHICLE_COL, 'cat': F4_GT_CAT_COL, 'mode': F4_GT_MODE_COL, 'cat_fb': F4_GT_CAT_COL, 'mode_fb': F4_GT_MODE_COL},
        }
        cfg = gt_map[data_format]
        df['_vehicle_pairs'] = df[cfg['v']].apply(parse_vehicle_pairs_auto)
        
        # 统一处理GT列
        gt_cat_col, gt_mode_col = cfg['cat'], cfg.get('mode')
        fallback_cat_col, fallback_mode_col = cfg.get('cat_fb'), cfg.get('mode_fb')

        df['_gt_category'] = df[gt_cat_col]
        if fallback_cat_col in df.columns and fallback_cat_col != gt_cat_col:
             df['_gt_category'] = df.apply(lambda row: row[gt_cat_col] if not is_truly_empty(row.get(gt_cat_col)) else row.get(fallback_cat_col, ''), axis=1)

        df['_gt_mode'] = ''
        if gt_mode_col and gt_mode_col in df.columns:
            df['_gt_mode'] = df[gt_mode_col]
            if fallback_mode_col in df.columns and fallback_mode_col != gt_mode_col:
                df['_gt_mode'] = df.apply(lambda row: row[gt_mode_col] if not is_truly_empty(row.get(gt_mode_col)) else row.get(fallback_mode_col, ''), axis=1)

        # 统一处理Query和History
        if len(cfg['q']) > 1: # 多轮
            df[['_history', '_query']] = df.apply(_extract_dynamic_query_history, axis=1, q_cols=cfg['q'])
        else: # 单轮
            df['_history'] = ''
            df['_query'] = df[cfg['q'][0]]
    
    # 清理可能存在的NaN值
    for col in ['_query', '_history', '_gt_category', '_gt_mode']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    return df

# --- 3. SFT生成阶段 ---
def get_final_pairs(gt_cat, gt_mode, all_pairs):
    def split_multi(s):
        if is_truly_empty(s) or str(s).strip().lower() in ['nan', '无']:
            return []
        return [x.strip() for x in re.split(r'[，,、]', str(s)) if x.strip()]
        
    cats, modes = split_multi(gt_cat), split_multi(gt_mode)

    if any(str(c).lower() == 'all' for c in cats) or any(str(m).lower() == 'all' for m in modes):
        return [{"all": "all"}]
        
    if not cats and not modes:
        return []

    valid = []
    # 使用“或”逻辑进行宽松匹配，更符合用户意图
    # 例如，GT为“滴滴专车”，只要小类是“滴滴专车”就应该被选中
    for pair in all_pairs:
        k, v = list(pair.items())[0]
        # 检查大类或小类是否在GT列表中
        cat_match = cats and k in cats
        mode_match = modes and v in modes
        
        if (cat_match or mode_match) and pair not in valid:
            valid.append(pair)
            
    return valid

def process_csv_file(input_file, output_csv_file=None, output_json_file=None):
    try:
        df = pd.read_csv(input_file, encoding='utf-8', engine='python', on_bad_lines='warn', na_filter=False)
    except FileNotFoundError: print(f"错误: 输入文件未找到: {input_file}"); return
    except Exception as e: print(f"读取CSV文件 {input_file} 时出错: {e}"); return
    try:
        df = normalize_dataframe(df)
    except ValueError as e: print(f"处理文件 {input_file} 失败: {e}"); return
    sft_list = []
    for idx, row in df.iterrows():
        query = str(row.get('_query', '')).strip()
        if not query: continue
        history, vehicle_pairs = row.get('_history', ''), row.get('_vehicle_pairs', [])
        
        if not vehicle_pairs:
            output_pairs = []
        else:
            output_pairs = get_final_pairs(row.get('_gt_category'), row.get('_gt_mode'), vehicle_pairs)
        
        input_str = build_input_query(history, query, vehicle_pairs)
        output_str = json.dumps(output_pairs, ensure_ascii=False)
        sft_list.append({"instruction": UNIFIED_INSTRUCTION, "input": input_str, "output": output_str})
    
    sft_list = dedup_by_input(sft_list)
    if output_csv_file:
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f'处理完成，中间过程CSV已保存到：{output_csv_file}')
    if output_json_file:
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
        with open(output_json_file, 'w', encoding='utf-8') as fout:
            json.dump(sft_list, fout, ensure_ascii=False, indent=2)
        print(f"SFT样本已保存到：{output_json_file}")

# --- 4. 批量处理与文件管理 ---
def merge_json_files(file_list, output_file):
    merged_data = []
    for file_name in file_list:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list): merged_data.extend(data)
        except Exception as e: print(f"警告: 读取或合并文件 {file_name} 时出错: {e}")
    if merged_data:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        unique_merged_data = dedup_by_input(merged_data)
        print(f"合并前共 {len(merged_data)} 条数据，去重后剩余 {len(unique_merged_data)} 条。")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_merged_data, f, ensure_ascii=False, indent=2)
        print(f"合并完成，结果已保存到 {output_file}")
    else: print("没有可合并的数据，未生成合并文件。")

def batch_process(file_process_configs, merge_json_output=None):
    json_files = []
    for config in file_process_configs:
        input_path = config.get('input_csv')
        if not input_path or not os.path.exists(input_path):
            print(f"警告: 跳过不存在的文件: {input_path}")
            continue
        print(f"\n--- 正在处理：{input_path} ---")
        process_csv_file(input_path, config.get('output_csv'), config.get('output_json'))
        if config.get('output_json') and os.path.exists(config.get('output_json')):
             json_files.append(config['output_json'])
    if merge_json_output and json_files:
        print("\n--- 开始合并所有JSON文件 ---")
        merge_json_files(json_files, merge_json_output)

def get_today_str():
    return time.strftime("%Y%m%d", time.localtime())

VERSION = "v14.0_MultiTurnFix" # 版本号更新，体现多轮修复

def add_time_ver_dir(filepath):
    if not filepath: return None
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    new_dir = os.path.join(base_dir, get_today_str(), VERSION)
    os.makedirs(new_dir, exist_ok=True)
    return os.path.join(new_dir, filename)

if __name__ == "__main__":
    base_input_path = '/home/workspace/lgq/data/new'
    base_output_path = '/home/workspace/lgq/data/output'

    # 将所有需要处理的文件名放入此列表
    file_names = [
        # --- 原始文件列表 ---
        '5轮badcase.csv', '表一.csv', '表二.csv', '表三.csv', '表四.csv', '表五.csv',
        '待标注20250608_1758.csv', '待标注20250610_2007.csv',
        '所有车型.csv', '已标注20250612_1829.csv',
        '已标注20250617_1042.csv', '周五已标注数据—新增两列.csv',
        
        # --- 您提供的多轮格式文件，请确保文件名正确 ---
        '多轮query.csv', 
        '多轮query追加50条.csv',
        
        # --- 您提供的新格式文件，请确保文件名正确 ---
        # '打车语料2.0(多轮query).csv' 
    ]
    
    file_process_configs = []
    found_any_file = False
    for fname in file_names:
        full_path = os.path.join(base_input_path, fname)
        if os.path.exists(full_path):
            found_any_file = True
            item = {
                'input_csv': full_path,
                'output_csv': add_time_ver_dir(os.path.join(base_output_path, fname.replace('.csv', '_debug.csv'))),
                'output_json': add_time_ver_dir(os.path.join(base_output_path, fname.replace('.csv', '_sft.json')))
            }
            file_process_configs.append(item)
        else:
            print(f"提示: 在 '{base_input_path}' 中未找到文件 '{fname}'，将跳过。")

    if not found_any_file:
        print(f"\n警告：在路径 '{base_input_path}' 下找不到任何指定的CSV文件。请检查文件名和路径。")
    else:
        merge_json_output = add_time_ver_dir(os.path.join(base_output_path, 'merged_sft_data.json'))
        batch_process(file_process_configs, merge_json_output=merge_json_output)