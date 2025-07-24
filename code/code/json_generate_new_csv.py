import pandas as pd
import json
import os

# --- 1. 定义固定的、详细的 instruction (系统提示词) ---
# 这部分内容直接从你提供的目标 JSON 样式中复制而来
instruction_prompt = """
你是一个指令解析专家，将用户的口语化指令转换为结构化的书面任务。

【任务】
根据用户输入，生成一个或多个任务计划。

【任务类型】
每个任务必须属于以下类型之一：
[订单, 发票, 购物车, 客服, 签到, 收藏, 搜索, 物流, 启动]

【应用名称】
- 标准应用名列表：[抖音, 抖音极速版, 快手, 快手极速版, 拼多多, 淘宝, 京东, 天猫, 闲鱼, 抖音火山版, 阿里巴巴, 唯品会, 得物, 转转]
- 将别名（如“毒”）或口语化名称（如“拼夕夕”）转换为标准名（如“得物”，“拼多多”）。

【输出格式】
严格遵循以下格式，使用<br>分隔多个任务。
<Plan>#任务类型#在$应用名称$中<任务内容>
#任务类型#打开$应用名称$</Plan><Finish>

【核心规则】
1.  **动词规范**：使用“打开、查看、搜索、管理、添加、使用、清理”等标准动词。
2.  **信息清理**：忽略用户输入中的所有无关词（如“YOYO”、“麻烦你”、“呢”、“啊”等）。
3.  **任务合并**：如果多个操作可以一步完成，则合并为一个任务（例如“打开淘宝购物车”是一个任务，不拆分）。只在任务完全独立时才拆分。
4.  **不创造任务**：不要猜测用户意图，只转换明确提到的指令。

【示例】
- 用户输入：YOYO，帮我看一下淘宝上有啥夏装卖
- 输出：<Plan>#搜索#在$淘宝$中查看夏装</Plan><Finish>

- 用户输入：麻烦你帮我打开拼夕夕购物车，然后再打开京东
- 输出：<Plan>#购物车#在$拼多多$中打开购物车<br>#启动#打开$京东$</Plan><Finish>

【用户输入】
"""

# --- 2. 定义文件路径 ---
# 请确保 CSV 文件与此脚本在同一目录下，或者提供完整路径
# csv_file_path = r"D:\Agent\data\全部生成数据表格 - 0707-500条指令-各5个query.csv"  # <--- 修改成你的 CSV 文件名
csv_file_path = r"D:\Agent\data\全部终版数据表格 - 0709-500条指令-额外各2个query.csv"  # <--- 修改成你的 CSV 文件名
json_output_path = r"D:\Agent\data\20250716_sft_formatted_dataset.json" # <--- 定义输出的 JSON 文件名

# --- 3. 数据处理 ---
try:
    # 读取 CSV 文件
    # 使用 encoding='utf-8-sig' 来处理可能存在的 BOM (字节顺序标记)
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    # 检查必要的列是否存在
    required_columns = ['final_query', '模型输出']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 文件缺少必要的列。需要: {required_columns}, 实际存在: {df.columns.tolist()}")

    # 初始化一个列表来存储所有的数据条目
    data_list = []

    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        # 获取 input 和 output
        # 使用 .strip() 清除可能存在的前后多余空格
        input_text = str(row['final_query']).strip()
        output_text = str(row['模型输出']).strip()
        
        # 检查数据是否为空，如果为空则跳过该行
        if not input_text or not output_text:
            print(f"警告：第 {index + 2} 行的 'final_query' 或 '模型输出' 为空，已跳过。")
            continue

        # 创建符合目标格式的字典
        entry = {
            "instruction": instruction_prompt.strip(),  # 使用 strip() 去除 instruction 的首尾空白
            "input": input_text,
            "output": output_text
        }
        
        # 将字典添加到列表中
        data_list.append(entry)

    # --- 4. 保存为 JSON 文件 ---
    # 使用 'w' 模式写入文件，ensure_ascii=False 确保中文字符正常显示
    # indent=4 使 JSON 文件格式化，更易读
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"成功处理 {len(data_list)} 条数据。")
    print(f"数据已成功保存到: {os.path.abspath(json_output_path)}")

except FileNotFoundError:
    print(f"错误：找不到文件 '{csv_file_path}'。请检查文件名和路径是否正确。")
except Exception as e:
    print(f"处理过程中发生错误: {e}")