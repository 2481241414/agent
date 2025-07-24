import json
import random
import os

# ==============================================================================
# 配置区域: 请根据您的需求修改这些变量
# ==============================================================================

# 输入的原始 JSON 文件名
INPUT_JSON_PATH = "/home/workspace/lgq/distill/data/20250714_sft_train_dataset.json" 

# 输出文件的保存目录
OUTPUT_DIR = "data"

# 输出的训练集和验证集文件名
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "/home/workspace/lgq/distill/data/20250714_sft_instruction_train.jsonl")
VAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "/home/workspace/lgq/distill/data/20250714_sft_instruction_val.jsonl")

# 训练集所占的比例 (例如 0.9 代表 90% 的数据用于训练)
SPLIT_RATIO = 0.9

# 随机种子，确保每次划分的结果都一样，便于复现实验
RANDOM_SEED = 42

# ==============================================================================
# 脚本主逻辑: 一般无需修改以下内容
# ==============================================================================

def write_jsonl(file_path: str, data: list):
    """将数据列表写入到 .jsonl 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # ensure_ascii=False 保证中文字符能被正确写入
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')

def main():
    """主执行函数"""
    print("脚本开始执行...")

    # 1. 检查并创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        print(f"创建输出目录: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # 2. 读取原始 JSON 文件
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
        print(f"成功从 '{INPUT_JSON_PATH}' 加载 {len(full_dataset)} 条数据。")
    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_JSON_PATH}' 不存在。请确保文件名正确。")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{INPUT_JSON_PATH}' 不是有效的 JSON 格式。")
        return

    # 3. 设置随机种子并打乱数据集
    random.seed(RANDOM_SEED)
    random.shuffle(full_dataset)
    print(f"数据集已使用随机种子 {RANDOM_SEED} 进行打乱。")

    # 4. 计算切分点
    total_records = len(full_dataset)
    train_size = int(total_records * SPLIT_RATIO)
    val_size = total_records - train_size

    if train_size == 0 or val_size == 0:
        print("警告: 数据集太小或切分比例不当，导致训练集或验证集为空。")
        print(f"总数据量: {total_records}, 训练集大小: {train_size}, 验证集大小: {val_size}")
        return

    # 5. 切分数据集
    train_data = full_dataset[:train_size]
    val_data = full_dataset[train_size:]
    print(f"数据切分完成：")
    print(f"  - 训练集: {len(train_data)} 条")
    print(f"  - 验证集: {len(val_data)} 条")

    # 6. 写入到 .jsonl 文件
    write_jsonl(TRAIN_OUTPUT_PATH, train_data)
    print(f"训练集已保存到: '{TRAIN_OUTPUT_PATH}'")
    
    write_jsonl(VAL_OUTPUT_PATH, val_data)
    print(f"验证集已保存到: '{VAL_OUTPUT_PATH}'")

    print("脚本执行完毕！")


if __name__ == "__main__":
    main()