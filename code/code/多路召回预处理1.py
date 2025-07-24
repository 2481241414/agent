import pandas as pd
import json
import csv
import sys

def create_instruction_to_tool_map(mapping_filename: str) -> dict:
    """
    从CSV文件解析工具映射表，创建一个从 instruction_template 到 function_name 的字典。

    Args:
        mapping_filename: 包含工具映射关系的CSV文件名。

    Returns:
        一个字典，键是 instruction_template, 值是对应的 function_name。
        如果文件未找到或无法处理，则返回 None。
    """
    mapping_dict = {}
    print(f"正在读取工具映射文件: {mapping_filename}...")
    try:
        with open(mapping_filename, mode='r', encoding='utf-8-sig') as f:
            # 使用 csv.DictReader 可以通过列名直接访问，更清晰
            reader = csv.DictReader(f)
            for row in reader:
                function_name = row.get('function_name', '').strip()
                instructions_str = row.get('包含的指令', '')

                # 如果没有function_name，则跳过此行
                if not function_name:
                    continue

                # 特殊处理 '启动' 类别
                if function_name == 'open_app(app)' and not instructions_str:
                    app_name = row.get('app', '').strip()
                    if app_name:
                        instruction_template = f"打开{app_name}"
                        mapping_dict[instruction_template] = function_name
                    continue

                # 解析指令列表
                if instructions_str:
                    try:
                        instructions = json.loads(instructions_str)
                        for instruction in instructions:
                            mapping_dict[instruction.strip()] = function_name
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析JSON指令列表 '{instructions_str}' 在文件 '{mapping_filename}' 中。跳过此条目。", file=sys.stderr)
                        continue
    
    except FileNotFoundError:
        print(f"错误: 映射文件 '{mapping_filename}' 未找到。请确保该文件存在于脚本的同一目录。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"读取或解析映射文件时发生未知错误: {e}", file=sys.stderr)
        return None
        
    print(f"工具映射表成功创建，共包含 {len(mapping_dict)} 条指令映射。")
    return mapping_dict

def process_data(input_filename: str, output_filename: str, instruction_map: dict):
    """
    读取源CSV文件，根据指令模板匹配正确的工具，并生成新的CSV文件。

    Args:
        input_filename: 输入的CSV文件名 (e.g., 'corrected_data.csv')。
        output_filename: 输出的CSV文件名 (e.g., '单gt.csv')。
        instruction_map: 指令到工具的映射字典。
    """
    if instruction_map is None:
        print("处理中止，因为工具映射表未能成功加载。", file=sys.stderr)
        return

    print(f"正在处理输入文件: {input_filename}...")
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_filename}' 未找到。", file=sys.stderr)
        return

    total_rows = len(df)
    processed_count = 0
    
    # 定义一个函数来处理每一行，这样可以应用到整个DataFrame上，效率更高
    def get_single_tool(row):
        nonlocal processed_count
        instruction_template = row['instruction_template'].strip()
        
        # 1. 根据指令模板查找正确的工具名
        correct_tool_name = instruction_map.get(instruction_template)
        
        if not correct_tool_name:
            print(f"警告: 在第 {row.name + 2} 行，未找到指令 '{instruction_template}' 的映射。将保留原始工具。", file=sys.stderr)
            return row['available_tools']

        # 2. 解析原始的工具列表
        try:
            available_tools_list = json.loads(row['available_tools'])
            if not isinstance(available_tools_list, list):
                 raise TypeError("JSON内容不是一个列表")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"警告: 在第 {row.name + 2} 行，解析 'available_tools' 失败: {e}。将保留原始值。", file=sys.stderr)
            return row['available_tools']
            
        # 3. 在列表中找到正确的工具
        correct_tool_info = None
        for tool in available_tools_list:
            if isinstance(tool, dict) and tool.get('name', '').strip() == correct_tool_name:
                correct_tool_info = tool
                break
        
        if correct_tool_info:
            processed_count += 1
            # 4. 创建只包含正确工具的JSON字符串
            return json.dumps([correct_tool_info], ensure_ascii=False)
        else:
            print(f"警告: 在第 {row.name + 2} 行，指令 '{instruction_template}' 对应的工具 '{correct_tool_name}' 未在可用工具列表中找到。将保留原始工具。", file=sys.stderr)
            return row['available_tools']

    # 应用处理函数到 'available_tools' 列
    df['available_tools'] = df.apply(get_single_tool, axis=1)

    # 5. 保存结果到新文件
    try:
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print("\n--- 处理完成 ---")
        print(f"总共处理了 {total_rows} 行。")
        print(f"成功更新了 {processed_count} 行的 'available_tools' 列。")
        print(f"结果已保存到文件: {output_filename}")
    except Exception as e:
        print(f"错误: 无法写入输出文件 '{output_filename}': {e}", file=sys.stderr)

# --- 主程序入口 ---
if __name__ == "__main__":
    # 定义文件名
    MAPPING_FILE = 'D:\Agent\data\大类-工具映射关系表-0707-Cleaned.csv'
    INPUT_FILE = 'D:\Agent\data\corrected_data.csv'
    OUTPUT_FILE = 'D:\Agent\data\单gt.csv'

    # 步骤1: 创建指令到工具的映射字典
    instruction_tool_map = create_instruction_to_tool_map(MAPPING_FILE)

    # 步骤2: 处理数据并生成新文件
    process_data(INPUT_FILE, OUTPUT_FILE, instruction_tool_map)