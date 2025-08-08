import pandas as pd
import ast
import os

# --- 1. 静态数据定义 ---
# 将两个平台的完整车列表放在这里，以便进行反向查找
# 这些数据必须和您生成SFT样本时使用的数据完全一致
DIDI_VEHICLE_LIST = [
    {"index": "1", "category": "推荐", "carModel": "极速拼车"}, {"index": "2", "category": "推荐", "carModel": "特惠快车"},
    {"index": "3", "category": "推荐", "carModel": "滴滴快车"}, {"index": "4", "category": "拼车", "carModel": "特价拼车"},
    {"index": "5", "category": "拼车", "carModel": "极速拼车"}, {"index": "6", "category": "特价", "carModel": "花小猪外接版"},
    {"index": "7", "category": "特价", "carModel": "特惠快车"}, {"index": "8", "category": "快速", "carModel": "花小猪正价版"},
    {"index": "9", "category": "快速", "carModel": "曹操出行"}, {"index": "10", "category": "快速", "carModel": "特快出租车"},
    {"index": "11", "category": "快速", "carModel": "滴滴特快"}, {"index": "12", "category": "快速", "carModel": "出租车"},
    {"index": "13", "category": "快速", "carModel": "滴滴快车"}, {"index": "14", "category": "舒适", "carModel": "滴滴专车"},
    {"index": "15", "category": "舒适", "carModel": "滴滴豪华车"}, {"index": "16", "category": "六座", "carModel": "六座专车"},
    {"index": "17", "category": "更多", "carModel": "滴滴代驾"}, {"index": "18", "category": "更多", "carModel": "宠物出行"},
    {"index": "19", "category": "更多", "carModel": "滴滴包车"}, {"index": "20", "category": "送货", "carModel": "汽车快送"},
    {"index": "21", "category": "送货", "carModel": "跑腿"}, {"index": "22", "category": "特价", "carModel": "惊喜特价"},
    {"index": "23", "category": "推荐", "carModel": "顺风车"}, {"index": "24", "category": "城际", "carModel": "城际拼车"},
    {"index": "25", "category": "城际", "carModel": "顺风车"}, {"index": "26", "category": "推荐", "carModel": "花小猪外接版"},
    {"index": "27", "category": "推荐", "carModel": "花小猪正价版"}, {"index": "28", "category": "推荐", "carModel": "滴滴专车"},
    {"index": "29", "category": "推荐", "carModel": "惊喜特价"}
]
GAODE_VEHICLE_LIST = [
    {"index": "1", "category": "拼车", "carModel": "极速拼车"}, {"index": "2", "category": "特惠快车", "carModel": "特惠快车"},
    {"index": "3", "category": "经济型", "carModel": "曹操出行"}, {"index": "4", "category": "经济型", "carModel": "T3出行"},
    {"index": "5", "category": "经济型", "carModel": "享道出行"}, {"index": "6", "category": "经济型", "carModel": "鞍马出行"},
    {"index": "7", "category": "经济型", "carModel": "腾飞出行"}, {"index": "8", "category": "经济型", "carModel": "及时用车"},
    {"index": "9", "category": "经济型", "carModel": "风韵出行"}, {"index": "10", "category": "经济型", "carModel": "环旅出行"},
    {"index": "11", "category": "经济型", "carModel": "神州专车"}, {"index": "12", "category": "经济型", "carModel": "叮叮出行"},
    {"index": "13", "category": "经济型", "carModel": "迪尔出行"}, {"index": "14", "category": "经济型", "carModel": "旅程约车"},
    # ... (为了简洁，省略了高德列表的剩余部分，但实际使用时请确保它是完整的)
    # 假设这里是完整的高德列表
]


# --- 2. 辅助函数 ---
def create_reverse_map(vehicle_list):
    """根据车列表，创建一个 (category, carModel) -> [index] 的反向映射。"""
    reverse_map = {}
    for item in vehicle_list:
        key = (item['category'], item['carModel'])
        if key not in reverse_map:
            reverse_map[key] = []
        reverse_map[key].append(item['index'])
    return reverse_map

# 创建全局的反向映射表
DIDI_REVERSE_MAP = create_reverse_map(DIDI_VEHICLE_LIST)
GAODE_REVERSE_MAP = create_reverse_map(GAODE_VEHICLE_LIST)

def get_app_name_from_options(option_str):
    """根据可选车型组合的字符串内容，判断是滴滴还是高德。"""
    if pd.isna(option_str):
        return "didi" # 默认值
    # 使用一些独特的车型来判断
    if "'T3出行'" in option_str or "'高德秒送'" in option_str:
        return "gaode"
    if "'滴滴快车'" in option_str or "'滴滴代驾'" in option_str:
        return "didi"
    # 默认或无法判断时返回滴滴
    return "didi"

def convert_to_indexed_format(combo_str, reverse_map):
    """将车型组合列表字符串，转换为带索引的格式字符串。"""
    if pd.isna(combo_str) or not combo_str:
        return "[]"
    
    try:
        combo_list = ast.literal_eval(combo_str)
        if not isinstance(combo_list, list):
            return "[]"
    except (ValueError, SyntaxError):
        # 如果解析失败，可能是一个不规范的字符串
        return f"[ERROR: Cannot parse '{combo_str}']"

    indexed_list = []
    for pair_dict in combo_list:
        if not isinstance(pair_dict, dict) or not pair_dict:
            continue
        
        try:
            category, car_model = list(pair_dict.items())[0]
            key = (category, car_model)
            
            # 从反向映射中查找索引
            indices = reverse_map.get(key)
            if indices:
                # 一个(category, carModel)可能对应多个index，全部加入
                for index in indices:
                    indexed_list.append({index: pair_dict})
            else:
                # 如果找不到，可以记录下来
                indexed_list.append({f"NOT_FOUND": pair_dict})

        except (IndexError, TypeError):
            continue

    if not indexed_list:
        return "[]"

    # 为了保持输出一致性，按 index (如果是数字) 排序
    indexed_list.sort(key=lambda x: int(list(x.keys())[0]) if list(x.keys())[0].isdigit() else 9999)
    
    return str(indexed_list)


# --- 3. 主处理函数 ---
def process_inference_file(input_path, output_path):
    """
    读取推理结果CSV，添加带索引的列，并保存到新文件。
    """
    print(f"正在读取文件: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {input_path}")
        return

    # 应用转换函数到每一行
    # 我们使用 apply 和 lambda 函数来处理每一行数据
    def transform_row(row):
        # 1. 判断APP类型
        option_combo_str = row['可选车型组合']
        app_name = get_app_name_from_options(option_combo_str)
        reverse_map_to_use = DIDI_REVERSE_MAP if app_name == "didi" else GAODE_REVERSE_MAP
        
        # 2. 转换模型输出
        model_combo_str = row['模型输出车型组合']
        model_indexed = convert_to_indexed_format(model_combo_str, reverse_map_to_use)
        
        # 3. 转换正确答案
        true_combo_str = row['正确车型组合']
        true_indexed = convert_to_indexed_format(true_combo_str, reverse_map_to_use)
        
        return model_indexed, true_indexed

    # apply函数会返回一个包含元组的Series，我们将其展开为两个新列
    df[['模型输出_indexed', '正确答案_indexed']] = df.apply(transform_row, axis=1, result_type='expand')

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 保存结果
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"处理完成！结果已保存到: {output_path}")


# --- 4. 执行 ---
if __name__ == "__main__":
    # ##############################################
    # #  请在这里修改你的输入和输出文件路径         #
    # ##############################################
    
    # 你的原始推理结果文件
    input_csv_path = '/home/workspace/lgq/data/inference/20250625/v2.0/打车语料-2.0版-624-zhao - 车型理解_inference_results_0625.csv'  
    
    # 你希望保存的新文件的路径
    output_csv_path = '/home/workspace/lgq/data/inference/20250625/v2.0/打车语料-2.0版-624-zhao - 车型理解_inference_results_0625_inference_results_indexed.csv'
    
    # ##############################################
    
    # 检查输入文件是否存在
    if not os.path.exists(input_csv_path):
        print(f"输入文件 '{input_csv_path}' 不存在。请检查路径或将你的文件重命名为 'inference_results.csv' 并放在脚本同级目录下。")
        # 为了方便演示，创建一个示例输入文件
        print("正在创建一个示例输入文件 'inference_results.csv'...")
        dummy_data = {
            'query': [
                '从北京西站打车去朝阳大悦城开会，记得要推荐类车型',
                '听说外滩夜景不错，给我打辆拼车服务从上海虹桥机场出发',
                '我用高德打车，要T3出行'
            ],
            '可选车型组合': [
                "[{'六座': '六座专车'}, {'城际': '城际拼车'}, {'快速': '滴滴快车'}, {'拼车': '极速拼车'}, {'推荐': '滴滴快车'}]",
                "[{'六座': '六座专车'}, {'拼车': '特价拼车'}, {'拼车': '极速拼车'}, {'推荐': '极速拼车'}]",
                "[{'经济型': 'T3出行'}, {'优享型': 'T3出行'}]"
            ],
            '模型输出车型组合': [
                "[{'推荐': '滴滴快车'}]",
                "[{'拼车': '特价拼车'}, {'拼车': '极速拼车'}]",
                "[{'经济型': 'T3出行'}]"
            ],
            '正确车型组合': [
                "[{'推荐': '滴滴快车'}, {'快速': '滴滴快车'}]",
                "[{'拼车': '特价拼车'}, {'拼车': '极速拼车'}]",
                "[{'经济型': 'T3出行'}, {'优享型': 'T3出行'}]"
            ]
        }
        pd.DataFrame(dummy_data).to_csv(input_csv_path, index=False, encoding='utf-8-sig')
    
    process_inference_file(input_csv_path, output_csv_path)