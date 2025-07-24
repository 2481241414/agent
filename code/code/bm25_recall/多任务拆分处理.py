import pandas as pd
import os

def process_all_multitask_csv_final(input_path, output_path):
    """
    读取CSV文件，稳健地处理所有多任务行（即使各列元素数量不匹配），
    将其展开为多个单任务行，并保持原始'类型'列的值不变。

    Args:
        input_path (str): 输入CSV文件的路径。
        output_path (str): 输出CSV文件的路径。
    """
    # --- 步骤 1: 检查并读取CSV文件 ---
    if not os.path.exists(input_path):
        print(f"错误：输入文件 '{input_path}' 不存在。请确保文件名和路径正确。")
        return

    print(f"正在读取文件: {input_path}")
    
    try:
        # 使用 fillna('') 在读取时就将空值替换为空字符串
        df = pd.read_csv(input_path, header=0, engine='python').fillna('')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
        
    # --- 步骤 2: 识别需要处理的行 ---
    # 通过判断'plan'列是否包含换行符来识别所有多任务行
    is_multi_task = df['plan'].astype(str).str.contains('\n', na=False)
    
    df_single_task = df[~is_multi_task]
    df_multi_task = df[is_multi_task]

    if df_multi_task.empty:
        print("文件中未找到需要展开的多任务行。")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已将原始数据直接保存至 '{output_path}'")
        return

    print(f"找到 {len(df_single_task)} 行单任务数据和 {len(df_multi_task)} 行需要展开的多任务数据。")

    # --- 步骤 3: 稳健地逐行处理多任务数据 ---
    new_rows = []
    columns_to_split = ['plan', 'reply', 'app', '大类', 'origin_instruct']

    for index, row in df_multi_task.iterrows():
        # 以 'plan' 列为基准进行拆分，它的长度决定了要生成多少新行
        base_split = str(row['plan']).strip().split('\n')
        num_tasks = len(base_split)

        # 拆分其他所有需要处理的列
        other_splits = {}
        for col in columns_to_split:
            if col != 'plan':
                other_splits[col] = str(row[col]).strip().split('\n')

        # 生成新行
        for i in range(num_tasks):
            new_row = row.to_dict()
            
            # 填充 plan
            new_row['plan'] = base_split[i].strip()

            # 安全地填充其他列，如果索引不存在则填充空字符串
            for col, splits in other_splits.items():
                try:
                    new_row[col] = splits[i].strip()
                except IndexError:
                    # 如果其他列的元素数量少于plan列，用空字符串填充
                    new_row[col] = ''
            
            # **核心改动：不再修改 '类型' 列的值**
            # 下面这几行修改类型的代码已被移除或注释掉
            # current_type = new_row['类型']
            # if '-多任务' in current_type:
            #     new_row['类型'] = current_type.replace('-多任务', '-单任务')
            # elif '-三任务' in current_type:
            #      new_row['类型'] = current_type.replace('-三任务', '-单任务')

            new_rows.append(new_row)

    # 将新行列表转换为 DataFrame
    if new_rows:
        df_expanded = pd.DataFrame(new_rows)
    else:
        # 如果没有任何多任务行被处理，创建一个空的DataFrame以避免后续错误
        df_expanded = pd.DataFrame(columns=df.columns)


    # --- 步骤 4: 合并数据并保存 ---
    # 确保两个DataFrame的列顺序一致再合并
    final_df = pd.concat([df_single_task, df_expanded], ignore_index=True, sort=False)
    final_df.fillna('', inplace=True)
    
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"🎉 数据处理成功！")
    print(f"总共生成 {len(final_df)} 行规整数据。")
    print(f"结果已保存到文件: '{output_path}'")

# --- 如何运行本脚本 ---
if __name__ == "__main__":
    # 请确保这里的路径是正确的
    input_file_path = r'D:\Agent\data\plan数据集 - 评估集.csv'  
    output_file_path = r'D:\Agent\code\bm25_recall\data\plan数据集 - 评估集_processed.csv'

    process_all_multitask_csv_final(input_file_path, output_file_path)