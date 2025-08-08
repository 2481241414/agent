import json
import random
import math
import os

# --- 可配置参数 ---
INPUT_FILE = '/home/workspace/lgq/data/output/20250625/v18.0_GTFallbackFix/merged_sft_data.json'   # 原始数据文件名
TRAIN_FILE = '/home/workspace/lgq/data/output/20250625/v18.0_GTFallbackFix/train_dataset_95.json'  # 输出的训练集文件名
TEST_FILE = '/home/workspace/lgq/data/output/20250625/v18.0_GTFallbackFix/test_dataset_05.json'   # 输出的测试集文件名
TEST_RATIO = 0.05                  # 测试集所占的比例 (例如 0.2 表示 20%)
RANDOM_SEED = 42                  # 随机种子，保证每次划分结果一致，便于复现

# --- 主逻辑 ---

def split_dataset_manual():
    """
    从JSON文件读取数据，手动进行划分，并保存到新文件。
    """
    # 1. 检查并读取原始数据文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 '{INPUT_FILE}'。请确保文件存在于当前目录。")
        return

    print(f"正在从 '{INPUT_FILE}' 读取数据...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            original_dataset = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：'{INPUT_FILE}' 文件格式不是有效的JSON。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return

    if not isinstance(original_dataset, list) or not original_dataset:
        print("错误：JSON文件中的数据不是一个有效的列表或列表为空。")
        return

    print(f"原始数据集总数: {len(original_dataset)}")

    # 2. 手动进行划分
    # 设置随机种子以保证结果可复现
    random.seed(RANDOM_SEED)

    # 复制并打乱数据集
    shuffled_data = original_dataset.copy()
    random.shuffle(shuffled_data)

    # 计算切分点
    split_index = math.ceil(len(shuffled_data) * (1 - TEST_RATIO))

    # 切分数据集
    train_data = shuffled_data[:split_index]
    test_data = shuffled_data[split_index:]
    
    print(f"正在按 {1-TEST_RATIO:.0%}/{TEST_RATIO:.0%} 的比例进行划分...")
    print(f"划分完成。训练集数量: {len(train_data)}, 测试集数量: {len(test_data)}")

    # 3. 将划分好的数据保存到新文件
    try:
        # 保存训练集
        with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 确保中文正常显示，indent=2 格式化输出
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"训练集已成功保存到 '{TRAIN_FILE}'")

        # 保存测试集
        with open(TEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"测试集已成功保存到 '{TEST_FILE}'")
        
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

# 运行主函数
if __name__ == '__main__':
    split_dataset_manual()













# import pandas as pd
# import numpy as np

# # 读取csv数据集
# df = pd.read_csv('/home/workspace/lgq/data/全新训练数据20250613.csv')

# # 随机打乱数据
# df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # 按5:95比例划分
# n_total = len(df_shuffled)
# n_part1 = int(n_total * 0.05)
# n_part2 = n_total - n_part1

# df_part1 = df_shuffled.iloc[:n_part1]
# df_part2 = df_shuffled.iloc[n_part1:]

# # 保存为新的csv文件
# df_part1.to_csv('/home/workspace/lgq/data/全新训练数据20250613_10percent.csv', index=False)
# df_part2.to_csv('/home/workspace/lgq/data/全新训练数据20250613_90percent.csv', index=False)
