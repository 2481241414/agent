import pandas as pd
import os
import json
import ast # <--- 关键改动：导入ast模块

def clean_mapping_file(input_path: str, output_path: str):
    """
    一个专门用于清洗和规范化 "大类-工具映射关系表" 的函数。
    此版本已修正对'包含的指令'列的解析逻辑。

    Args:
        input_path (str): 原始映射表文件的路径。
        output_path (str): 清洗后要保存的新文件的路径。
    """
    print(f"--- 开始清洗文件: {os.path.basename(input_path)} ---")

    try:
        df = pd.read_csv(input_path, engine='python', sep=',', dtype=str)
        print(f"原始文件读取成功，共 {len(df)} 行。")

        df['function_name'] = df['function_name'].str.strip('" ').str.strip()
        print("步骤 1/3: 已清洗 'function_name' 列。")

        df['包含指令数量'] = pd.to_numeric(df['包含指令数量'], errors='coerce').fillna(0).astype(int)
        print("步骤 2/3: 已处理 '包含指令数量' 列。")

        # --- 核心改动在这里 ---
        def normalize_instructions(cell_value):
            """
            使用 ast.literal_eval 安全地解析Python列表格式的字符串。
            """
            # 如果单元格是空的或不是字符串，直接返回空列表的JSON格式
            if pd.isna(cell_value) or not isinstance(cell_value, str) or cell_value.strip() == '':
                return '[]'
            
            clean_value = cell_value.strip()
            
            try:
                # 使用 ast.literal_eval 来解析字符串
                parsed_list = ast.literal_eval(clean_value)
                
                # 确保解析出来的是一个列表
                if isinstance(parsed_list, list):
                    # 使用json.dumps将其转换为标准的JSON数组字符串，以便于后续处理
                    return json.dumps(parsed_list, ensure_ascii=False)
                else:
                    # 如果解析出来不是列表（例如，只是一个普通字符串），则返回空列表
                    return '[]'
            except (ValueError, SyntaxError):
                # 如果ast.literal_eval解析失败，说明字符串格式不正确，返回空列表
                return '[]'

        df['包含的指令'] = df['包含的指令'].apply(normalize_instructions)
        print("步骤 3/3: 已使用正确的方法规范化 '包含的指令' 列。")

        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n🎉 清洗完成！已将规范化后的数据保存到:\n{output_path}")

        print("\n--- 清洗后数据预览 (前5行) ---")
        print(df.head().to_string())
        print("\n--- 清洗后数据预览 (随机5行) ---")
        print(df.sample(5).to_string())

    except FileNotFoundError:
        print(f"错误：输入文件未找到！请检查路径：{input_path}")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    original_mapping_path = r'D:\Agent\data\815版本-大类-工具关系映射表-0710指令集-0715 - 大类-app-func-0717修改指令名称.csv'
    cleaned_mapping_path = r'D:\Agent\data\大类-工具映射关系表-0815-Cleaned.csv'
    
    clean_mapping_file(input_path=original_mapping_path, output_path=cleaned_mapping_path)