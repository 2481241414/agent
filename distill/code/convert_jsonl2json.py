import json
import sys

def convert_jsonl_to_json_list(input_file_path: str, output_file_path: str):
    """
    读取一个包含 instruction/input/output 格式的 JSONL 文件，
    并将其转换为一个包含相同对象的标准 JSON 列表文件。
    
    这个脚本只改变文件结构（从 JSONL 到 JSON 列表），不改变任何数据内容。

    参数:
    - input_file_path (str): 输入的源 .jsonl 文件路径。
    - output_file_path (str): 输出的目标 .json 文件路径。
    """
    
    print(f"--- 开始转换文件 ---")
    print(f"  输入 (JSONL): {input_file_path}")
    print(f"  输出 (JSON):  {output_file_path}")

    # 用于存储所有原始数据对象的列表
    final_data_list = []
    
    processed_lines = 0
    skipped_lines = 0

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                try:
                    # 1. 解析原始的JSON对象
                    source_data = json.loads(stripped_line)
                    
                    # 2. 定义我们需要的键
                    required_keys = ['instruction', 'input', 'output']
                    
                    # 3. 检查所有需要的键是否存在
                    if not all(key in source_data for key in required_keys):
                        print(f"警告: 第 {line_num} 行缺少必要的键（instruction, input, 或 output），已跳过。", file=sys.stderr)
                        skipped_lines += 1
                        continue
                        
                    # 4. 创建一个新的字典，只包含这三个键，以确保没有多余字段
                    target_record = {
                        "instruction": source_data["instruction"],
                        "input": source_data["input"],
                        "output": source_data["output"]
                    }
                    
                    # 5. 将这个与源数据内容完全相同的字典添加到列表中
                    final_data_list.append(target_record)
                    processed_lines += 1

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"警告: 第 {line_num} 行解析失败或格式错误，错误: {e}。已跳过。", file=sys.stderr)
                    skipped_lines += 1
        
        # 6. 将最终的列表写入目标JSON文件
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # indent=2 使输出的JSON文件格式化，易于阅读
            json.dump(final_data_list, outfile, indent=2, ensure_ascii=False)
            
        print("\n--- 转换成功！ ---")
        print(f"  成功处理了 {processed_lines} 条记录。")
        if skipped_lines > 0:
            print(f"  跳过了 {skipped_lines} 条格式不兼容的记录。")
        print(f"  您的原始数据已按新结构保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"\n错误: 输入文件 '{input_file_path}' 未找到。", file=sys.stderr)
    except Exception as e:
        print(f"\n在处理过程中发生未知错误: {e}", file=sys.stderr)

# ==============================================================================
#  使用方法:
#  1. 将下面的文件名替换成你的实际文件。
#  2. 在终端中运行 `python your_script_name.py`。
# ==============================================================================
if __name__ == '__main__':
    # --- 请在这里配置您的文件路径 ---
    input_path = '/home/workspace/LLaMA-Factory/data/train_迭代5forlevel0badcase0801_merge_alpaca.jsonl'  # 您的原始数据文件
    output_path = '/home/workspace/LLaMA-Factory/data/train_迭代5forlevel0badcase0801_merge_alpaca.json' # 您希望生成的、结构正确的文件
    # --------------------------------

    # 调用转换函数
    convert_jsonl_to_json_list(input_path, output_path)


