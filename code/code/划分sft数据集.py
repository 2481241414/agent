import json
import random
import os

def split_dataset_in_chunks(input_file_path: str, train_output_path: str, test_output_path: str, chunk_size: int = 5):
    """
    Reads a JSON dataset, splits it into training and testing sets in chunks,
    and saves them to separate files.

    Args:
        input_file_path (str): Path to the input JSON file.
        train_output_path (str): Path to save the training set JSON file.
        test_output_path (str): Path to save the testing set JSON file.
        chunk_size (int): The size of each group to split from. Default is 5.
    """
    try:
        # --- 1. 读取原始数据集 ---
        with open(input_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list):
            raise TypeError("输入的 JSON 文件内容必须是一个列表 (list)。")
            
        print(f"成功读取数据集，总计 {len(dataset)} 条数据。")

        # --- 2. 初始化训练集和测试集列表 ---
        train_data = []
        test_data = []

        # --- 3. 按分组进行划分 ---
        # 使用 random.shuffle 可以打乱整个数据集，确保分组的随机性
        # 如果你希望保持原始顺序，可以注释掉下面这行
        # print("正在打乱数据集以确保随机性...")
        # random.shuffle(dataset)
        
        print(f"开始按 {chunk_size} 条数据为一组进行划分...")
        for i in range(0, len(dataset), chunk_size):
            # 获取当前分组
            chunk = dataset[i : i + chunk_size]
            
            # 如果当前分组的大小等于 chunk_size，则正常按比例划分
            if len(chunk) == chunk_size:
                # 生成当前分组的索引列表，例如 [0, 1, 2, 3, 4]
                indices = list(range(chunk_size))
                
                # 随机选择一个索引作为测试集样本的索引
                test_index = random.choice(indices)
                
                # 遍历当前分组，根据索引进行分配
                for j, item in enumerate(chunk):
                    if j == test_index:
                        test_data.append(item)
                    else:
                        train_data.append(item)
            else:
                # 如果是最后一个不足 chunk_size 的分组，全部划入训练集
                # 这样做可以避免丢失数据，并稍微增加训练数据量
                print(f"发现最后剩余 {len(chunk)} 条数据，全部划入训练集。")
                train_data.extend(chunk)

        # --- 4. 结果统计与输出 ---
        total_processed = len(train_data) + len(test_data)
        train_percentage = len(train_data) / total_processed * 100
        test_percentage = len(test_data) / total_processed * 100

        print("\n--- 数据划分完成 ---")
        print(f"训练集数据量: {len(train_data)} ({train_percentage:.2f}%)")
        print(f"测试集数据量: {len(test_data)} ({test_percentage:.2f}%)")
        print(f"总处理数据量: {total_processed}")

        # --- 5. 保存划分后的数据集 ---
        # 保存训练集
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        print(f"训练集已保存到: {os.path.abspath(train_output_path)}")

        # 保存测试集
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
        print(f"测试集已保存到: {os.path.abspath(test_output_path)}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file_path}'。请检查路径是否正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == '__main__':
    # --- 配置输入和输出文件路径 ---
    # 你的源文件名
    input_json_file = r'D:\Agent\data\20250714_sft_formatted_dataset.json'  # <--- 修改成你的 JSON 文件名
    
    # 定义输出的训练集和测试集文件名
    train_json_file = r'D:\Agent\data\20250714_sft_train_dataset.json'
    test_json_file = r'D:\Agent\data\20250714_sft_test_dataset.json'
    
    # 调用划分函数
    split_dataset_in_chunks(
        input_file_path=input_json_file,
        train_output_path=train_json_file,
        test_output_path=test_json_file,
        chunk_size=5  # 每 5 条数据为一组进行 4:1 划分
    )