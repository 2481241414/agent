import pandas as pd
import ast
import numpy as np

# --- 核心计算逻辑 (无改动) ---

def dict_list_to_set(data_list):
    if not isinstance(data_list, list):
        return set()
    return set(str(item) for item in data_list)

def get_prec_and_rec(ans, ground):
    ans = '' if pd.isna(ans) else ans
    ground = '' if pd.isna(ground) else ground
    try:
        ans_list = ast.literal_eval(ans) if ans and ans.lower() != 'nan' else []
        ground_list = ast.literal_eval(ground) if ground and ground.lower() != 'nan' else []
    except (ValueError, SyntaxError, TypeError):
        ans_list, ground_list = [], []

    ans_set = dict_list_to_set(ans_list)
    ground_set = dict_list_to_set(ground_list)
    intersect = len(ans_set & ground_set)

    if len(ans_set) == 0 and len(ground_set) == 0:
        return {'prec': 1.0, 'rec': 1.0}
    if len(ans_set) == 0 or len(ground_set) == 0:
        return {'prec': 0.0, 'rec': 0.0}
    if len(ans_set - ground_set) > 0:
        rec = intersect / len(ground_set)
        return {'prec': 0.0, 'rec': rec}
    prec = 1.0
    rec = intersect / len(ground_set)
    return {'prec': prec, 'rec': rec}

def process_csv_file(input_filename, output_filename):
    """
    读取CSV文件，计算指标，保存到新的Excel文件，并返回总体指标。
    """
    try:
        print(f"正在处理文件: {input_filename}...")
        df = pd.read_csv(input_filename)
        ans_col = '模型2返回结果'
        ground_col = '模型2正确结果'

        if ans_col not in df.columns or ground_col not in df.columns:
            print(f"错误: 文件 '{input_filename}' 中缺少必需的列 '{ans_col}' 或 '{ground_col}'。")
            return None

        metrics_series = df.apply(
            lambda row: get_prec_and_rec(row[ans_col], row[ground_col]),
            axis=1
        )
        metrics_df = metrics_series.apply(pd.Series)
        metrics_df = metrics_df.rename(columns={'prec': 'precision', 'rec': 'recall'})
        output_df = pd.concat([df, metrics_df[['precision', 'recall']]], axis=1)

        output_df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"处理完成！Excel结果已保存到: {output_filename}")
        
        # 计算并返回普通的平均指标
        avg_precision = metrics_df['precision'].mean()
        avg_recall = metrics_df['recall'].mean()
        
        return {'precision': avg_precision, 'recall': avg_recall}

    except FileNotFoundError:
        print(f"错误: 未找到文件 '{input_filename}'。请确保文件存在于脚本所在的目录中。")
        return None
    except Exception as e:
        print(f"处理文件 '{input_filename}' 时发生未知错误: {e}")
        return None

# --- 主程序 (只修改了打印文本) ---
if __name__ == "__main__":
    single_turn_input = '/home/workspace/lgq/data/待标注-20250627-1337 - 非多轮.csv'
    single_turn_output = '/home/workspace/lgq/data/single_turn_with_metrics.xlsx'
    
    multi_turn_input = '/home/workspace/lgq/data/待标注-20250627-1337 - 多轮.csv'
    multi_turn_output = '/home/workspace/lgq/data/multi_turn_with_metrics.xlsx'
    
    all_metrics = []

    print("--- 开始处理非多轮文件 ---")
    single_turn_metrics = process_csv_file(single_turn_input, single_turn_output)
    if single_turn_metrics:
        print(f"非多轮文件总体指标:")
        print(f"  - 平均精确率 (Average Precision): {single_turn_metrics['precision']:.4f}")
        print(f"  - 平均召回率 (Average Recall):    {single_turn_metrics['recall']:.4f}")
        all_metrics.append(single_turn_metrics)
    print("-" * 40)

    print("--- 开始处理多轮文件 ---")
    multi_turn_metrics = process_csv_file(multi_turn_input, multi_turn_output)
    if multi_turn_metrics:
        print(f"多轮文件总体指标:")
        print(f"  - 平均精确率 (Average Precision): {multi_turn_metrics['precision']:.4f}")
        print(f"  - 平均召回率 (Average Recall):    {multi_turn_metrics['recall']:.4f}")
        all_metrics.append(multi_turn_metrics)
    print("-" * 40)
    
    if len(all_metrics) > 0:
        total_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
        total_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
        print("--- 所有已处理文件的总平均指标 ---")
        print(f"  - 总平均精确率: {total_precision:.4f}")
        print(f"  - 总平均召回率: {total_recall:.4f}")
    else:
        print("没有文件被成功处理，无法计算总体指标。")