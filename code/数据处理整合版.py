import os
import re
import json
import csv
import ast
import time
import pandas as pd

def parse_vehicle_candidates_json(candidate_str):
    """
    解析多行JSON字符串形式的车辆候选列表，提取所有不重复的车型大类和车辆类型
    """
    categories = set()
    modes = set()
    if not isinstance(candidate_str, str):
        return [], []
    items = re.findall(r'\{.*?\}', candidate_str, re.DOTALL)
    for item in items:
        try:
            obj = json.loads(item.replace("'", '"'))
            if 'category' in obj and obj['category'] and obj['category'] != 'undefined':
                categories.add(obj['category'])
            if 'carModel' in obj and obj['carModel'] and obj['carModel'] != 'undefined':
                modes.add(obj['carModel'])
        except Exception:
            continue
    return list(categories), list(modes)

def parse_vehicle_candidates_markdown(candidate_str):
    """
    解析markdown表格形式的车辆候选列表，提取所有不重复的车型大类和车辆类型
    """
    if not isinstance(candidate_str, str):
        return [], []
    lines = candidate_str.split('\n')
    categories = set()
    modes = set()
    for line in lines:
        if not line or '车型大类' in line or '---' in line or '序号id' in line:
            continue
        fields = line.split('|')
        if len(fields) < 5:
            continue
        category = fields[3].strip()
        mode = fields[4].strip()
        if category and category != 'undefined':
            categories.add(category)
        if mode and mode != 'undefined':
            modes.add(mode)
    return list(categories), list(modes)

def detect_format(row):
    """
    检测车辆候选列表字段的格式：json、markdown或未知
    """
    candidate_str = row.get('车辆候选列表', '')
    if isinstance(candidate_str, float):
        return 'unknown'
    if re.search(r'^\s*\{.*\}\s*$', candidate_str.replace('\n', ''), re.DOTALL):
        return 'json'
    if '|' in candidate_str:
        return 'markdown'
    return 'unknown'

def parse_vehicle_candidates_auto(candidate_str):
    """
    自动判断格式并解析车辆候选列表
    """
    if not isinstance(candidate_str, str):
        return [], []
    if re.search(r'^\s*\{.*\}\s*$', candidate_str.replace('\n', ''), re.DOTALL):
        return parse_vehicle_candidates_json(candidate_str)
    elif '|' in candidate_str:
        return parse_vehicle_candidates_markdown(candidate_str)
    else:
        return [], []

def get_final_value(row, key, answer_key):
    """
    优先用正确答案，否则用原有字段
    """
    val = row.get(key, '')
    if pd.notna(row.get(answer_key, None)) and str(row[answer_key]).strip() not in ['', 'nan']:
        val = row[answer_key]
    return val

def parse_sft_field(field):
    """
    把最终category或最终mode内容转为list，如果为空返回[]
    """
    field = (str(field) or '').strip()
    if not field or field == '无':
        return []
    try:
        val = ast.literal_eval(field)
        if isinstance(val, list):
            return val
        elif isinstance(val, str):
            return [val]
        else:
            return []
    except Exception:
        return [field]

def parse_app(cheliang_candidate_str):
    if "高德地图" in cheliang_candidate_str:
        return "高德地图"
    elif "滴滴" in cheliang_candidate_str:
        return "滴滴"
    else:
        return "未知"

def build_instruction(app, query, cheliang_dalei_list, cheliang_xiaolei_list):
    cheliang_dalei_str = str(cheliang_dalei_list)
    cheliang_xiaolei_str = str(cheliang_xiaolei_list)
    if app == "高德地图":
        instr = f"""你是一个高德打车助手，你能根据用户的query去选择车型大类和车型小类，其中车型大类如["经济型"，“特快车”，“出租车”，“优享型”，“轻享专车”，“六座商务”，“豪华型”]等，车型小类如["特惠快车","曹操出行","5U出行","中军出行","斑马快跑","神州专车","飞嘀打车","T3出行","全民GO","妥妥E行","聚的新出租","阳光出行","聚优出租"]等。 

用户的query是：{query}
用户可以选择的车型大类为{cheliang_dalei_str}
用户可以选择的车型小类为{cheliang_xiaolei_str}

输出格式如下：
大类:[大类1,大类2]; 小类:[小类1,小类2]

特殊情况说明：
如果用户对车大类或者车小类没有明确要求，则大类和小类的字段都为空。 如query为：给我打个车去朱三家龙虾吃饭， 则输出大类:[]; 小类:[]
如果用户要求打所有车型，则输出所有的大类及所有的小类。 如query为：帮我打所有的车去火车站，则输出大类:["经济型"，“特快车”，“出租车”，“优享型”，“轻享专车”，“六座商务”，“豪华型”]; 小类:["远途极速拼","特惠快车","曹操出行","5U出行","中军出行","斑马快跑","东潮出行","神州专车","飞嘀打车","中交出行","T3出行","全民GO","妥妥E行","如祺出行","及时用车","享道出行","阳光出行","聚的新出租","深圳出租"]
当用户要求的车在可选列表中没有时，则输出大类:[]; 小类:[]，如:郑州东站到国际会展中心打车，要求七座MPV车型或者有婴儿座椅的车。则输出大类:[]; 小类:[]

部分策略参考：
用户要求“拼车”时，选择大类和小类中含有“拼车”的车
用户要求“出租”时，选择大类和小类中含有“出租”或者“的士”的车
用户要求“豪华”时，选择大类和小类中含有“豪华”的车
用户要求“快车”或者“快速车”时，“快速”的车都属于快车，例如：特快车、特惠快车、特快出租车、滴滴快车、滴滴特快等
用户要求“特价车”或者“经济型”时，特价拼车、特价出租车、特惠快车、经济型都属于这类
六座专车与六座豪华属于不同车型，要求打六座专车时不能返回六座豪华车

注意：
不要根据车辆的价格来选择车型，一定只能根据用户的语义来选择车型
满足条件的车大类和车小类都要选上，不能漏选，比如“出租”、“的士”都属于“出租车”
只能从用户可以选择的车辆详情信息中选择，不能自己造一些大类和小类，输出的内容必须和可选择的车辆详情内的内容完全一致"""
    else:
        instr = f"""你是一个滴滴打车助手，你能根据用户的query去选择车型大类和车型小类，其中车型大类如["拼车"，“特价”，“快速”，“舒适”，“六座”]等，车型小类如["特惠快车","曹操出行","滴滴快车","特快出租车","滴滴特快","出租车","滴滴专车","滴滴豪华车","六座专车","极速拼车","惊喜特价","花小猪外接版","花小猪正价版"]等。

用户的query是：{query}
用户可以选择的车型大类为{cheliang_dalei_str}
用户可以选择的车型小类为{cheliang_xiaolei_str}

输出格式如下：
大类:[大类1,大类2]; 小类:[小类1,小类2]

特殊情况说明：
如果用户对车大类或者车小类没有明确要求，则大类和小类的字段都为空。 如query为：给我打个车去朱三家龙虾吃饭， 则输出大类:[]; 小类:[]
如果用户要求打所有车型，则输出所有的大类及所有的小类。 如query为：帮我打所有的车去火车站，则输出大类:["拼车"，“特价”，“快速”，“舒适”，“六座”]; 小类:["特惠快车","曹操出行","滴滴快车","特快出租车","滴滴特快","出租车","滴滴专车","滴滴豪华车","六座专车","极速拼车","惊喜特价","花小猪外接版","花小猪正价版"]
当用户要求的车在可选列表中没有时，则输出大类:[]; 小类:[]，如:郑州东站到国际会展中心打车，要求七座MPV车型或者有婴儿座椅的车。则输出大类:[]; 小类:[]

部分策略参考：
用户要求“拼车”时，选择大类和小类中含有“拼车”的车
用户要求“出租”时，选择大类和小类中含有“出租”或者“的士”的车
用户要求“豪华”时，选择大类和小类中含有“豪华”的车
用户要求“快车”或者“快速车”时，“快速”的车都属于快车，例如：特快车、特惠快车、特快出租车、滴滴快车、滴滴特快等
用户要求“特价车”或者“经济型”时，特价拼车、特价出租车、特惠快车、经济型都属于这类
六座专车与六座豪华属于不同车型，要求打六座专车时不能返回六座豪华车

注意：
不要根据车辆的价格来选择车型，一定只能根据用户的语义来选择车型
满足条件的车大类和车小类都要选上，不能漏选，比如“出租”、“的士”都属于“出租车”
只能从用户可以选择的车辆详情信息中选择，不能自己造一些大类和小类，输出的内容必须和可选择的车辆详情内的内容完全一致"""
    return instr

def process_csv_file(input_file, output_csv_file=None, output_json_file=None):
    """
    处理单个csv文件，统一生成车型大类/小类列表、最终category/mode，并可生成SFT训练样本
    """
    df = pd.read_csv(input_file, encoding='utf-8', engine='python')
    categories_list = []
    modes_list = []
    final_category = []
    final_mode = []

    for idx, row in df.iterrows():
        cats, mods = parse_vehicle_candidates_auto(row['车辆候选列表'])
        categories_list.append(cats)
        modes_list.append(mods)
        cat_val = get_final_value(row, 'category', 'category正确答案')
        final_category.append(cat_val)
        mode_val = get_final_value(row, 'mode', 'mode正确答案')
        final_mode.append(mode_val)

    df['车型大类列表'] = categories_list
    df['车辆类型列表'] = modes_list
    df['最终category'] = final_category
    df['最终mode'] = final_mode

    if output_csv_file:
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f'处理完成，结果已保存到：{output_csv_file}')

    # 如果需要生成SFT样本
    if output_json_file:
        sft_list = []
        for _, row in df.iterrows():
            query = str(row.get('问题', '')).strip() or str(row.get('query', '')).strip()
            if not query:
                continue
            cheliang_candidate_str = row.get('车辆候选列表', '')
            dalei_str = row.get('车型大类列表', '[]')
            xiaolei_str = row.get('车辆类型列表', '[]')
            try:
                cheliang_dalei_list = ast.literal_eval(str(dalei_str))
            except Exception:
                cheliang_dalei_list = []
            try:
                cheliang_xiaolei_list = ast.literal_eval(str(xiaolei_str))
            except Exception:
                cheliang_xiaolei_list = []
            app = parse_app(cheliang_candidate_str)
            instruction = build_instruction(app, query, cheliang_dalei_list, cheliang_xiaolei_list)
            category_list = parse_sft_field(row.get('最终category', ''))
            mode_list = parse_sft_field(row.get('最终mode', ''))
            output_str = f"大类:{category_list}; 小类:{mode_list}"
            sft_sample = {
                "instruction": instruction,
                "input": "",
                "output": output_str
            }
            sft_list.append(sft_sample)
        with open(output_json_file, 'w', encoding='utf-8') as fout:
            json.dump(sft_list, fout, ensure_ascii=False, indent=2)
        print(f"已保存到 {output_json_file}，共生成 {len(sft_list)} 条SFT样本。")

def merge_json_files(file_list, output_file):
    """
    合并多个json文件（每个文件为list结构），保存为新的json文件
    """
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
    """
    批量处理多个csv文件，并可合并所有json输出
    file_process_configs: [
        {
            'input_csv': 'xxx.csv',
            'output_csv': 'xxx_out.csv',
            'output_json': 'xxx_out.json'
        },
        ...
    ]
    """
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

if __name__ == "__main__":
    # 你可以在这里配置需要批量处理的文件
    file_process_configs = [
        {
            'input_csv': '/home/workspace/lgq/data/表一.csv',
            'output_csv': '/home/workspace/lgq/data/表一output.csv',
            'output_json': '/home/workspace/lgq/data/markdown_sft_output.json'
        },
        {
            'input_csv': '/home/workspace/lgq/data/表二.csv',
            'output_csv': '/home/workspace/lgq/data/表二output.csv',
            'output_json': '/home/workspace/lgq/data/json_sft_output_2.json'
        },
        {
            'input_csv': '/home/workspace/lgq/data/表三.csv',
            'output_csv': '/home/workspace/lgq/data/表三output.csv',
            'output_json': '/home/workspace/lgq/data/json_sft_output_3.json'
        },
        {
            'input_csv': '/home/workspace/lgq/data/表四.csv',
            'output_csv': '/home/workspace/lgq/data/表四output.csv',
            'output_json': '/home/workspace/lgq/data/json_sft_output_4.json'
        },
        {
            'input_csv': '/home/workspace/lgq/data/表五.csv',
            'output_csv': '/home/workspace/lgq/data/表五output.csv',
            'output_json': '/home/workspace/lgq/data/json_sft_output_5.json'
        },
        # 可以继续添加其它csv的配置
    ]
    # 合并所有生成的json为一个大json
    merge_json_output = '/home/workspace/lgq/data/merge_data.json'
    batch_process(file_process_configs, merge_json_output)
