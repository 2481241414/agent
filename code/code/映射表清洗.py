import pandas as pd
import os
import json

def clean_mapping_file(input_path: str, output_path: str):
    """
    一个专门用于清洗和规范化 "大类-工具映射关系表" 的函数。

    Args:
        input_path (str): 原始映射表文件的路径。
        output_path (str): 清洗后要保存的新文件的路径。
    """
    print(f"--- 开始清洗文件: {os.path.basename(input_path)} ---")

    try:
        # 1. 读取原始CSV文件
        # 使用 engine='python' 和 sep=',' 来更好地处理不规范的CSV格式
        # dtype=str 确保所有列都先作为字符串读入，避免pandas自动转换类型导致问题
        df = pd.read_csv(input_path, engine='python', sep=',', dtype=str)
        print(f"原始文件读取成功，共 {len(df)} 行。")

        # 2. 清洗 'function_name' 列
        #    - .str.strip('" ') 会移除开头和结尾的双引号和空格
        #    - .str.strip() 会再次移除可能在引号内部的、靠边的空格
        df['function_name'] = df['function_name'].str.strip('" ').str.strip()
        print("步骤 1/3: 已清洗 'function_name' 列，移除了多余的引号和空格。")

        # 3. 填充 '包含指令数量' 列的缺失值
        #    - .fillna(0) 将所有 NaN (空值) 替换为 0
        #    - .astype(int) 将整列转换为整数类型
        df['包含指令数量'] = pd.to_numeric(df['包含指令数量'], errors='coerce').fillna(0).astype(int)
        print("步骤 2/3: 已处理 '包含指令数量' 列，空值已填充为0，并转为整数。")

        # 4. 规范化 '包含的指令' 列
        #    - 对于空值或非字符串值，我们填充一个表示空的JSON数组字符串 '[]'
        def normalize_instructions(cell_value):
            if pd.isna(cell_value) or not isinstance(cell_value, str) or cell_value.strip() == '':
                return '[]'
            # 确保它是一个合法的JSON数组格式
            # 移除可能存在的多余的外部双引号，然后重新用json.dumps包裹
            try:
                # 尝试解析，如果成功说明格式OK，重新格式化即可
                parsed_list = json.loads(cell_value)
                return json.dumps(parsed_list, ensure_ascii=False)
            except json.JSONDecodeError:
                # 如果解析失败，说明格式有问题，我们返回一个空列表字符串
                return '[]'

        df['包含的指令'] = df['包含的指令'].apply(normalize_instructions)
        print("步骤 3/3: 已规范化 '包含的指令' 列，空值已填充为 '[]'。")

        # 5. 保存到新文件
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n🎉 清洗完成！已将规范化后的数据保存到:\n{output_path}")

        # 6. 打印几行处理后的数据进行预览
        print("\n--- 清洗后数据预览 (前5行) ---")
        print(df.head().to_string())
        print("\n--- 清洗后数据预览 (最后5行，检查'启动'大类) ---")
        print(df.tail().to_string())


    except FileNotFoundError:
        print(f"错误：输入文件未找到！请检查路径：{input_path}")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 配置区域 ---
    # 您的原始、未处理的映射表文件路径
    # original_mapping_path = r'D:\Agent\data\大类-工具映射关系表-0707 - 大类-app-func.csv'
    original_mapping_path = r'D:\Agent\data\815版本-大类-工具关系映射表-0710指令集-0715 - 大类-app-func-0717修改指令名称.csv'
    
    # 定义清洗后要保存的新文件的路径
    # cleaned_mapping_path = r'D:\Agent\data\大类-工具映射关系表-0707-Cleaned.csv'
    cleaned_mapping_path = r'D:\Agent\data\大类-工具映射关系表-0815-Cleaned.csv'

    # 执行清洗函数
    clean_mapping_file(input_path=original_mapping_path, output_path=cleaned_mapping_path)