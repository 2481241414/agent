import os

# --- 配置 ---
# 请将 'input.jsonl' 替换为您的源文件名
# 假设您的源数据文件名是 query_response_pairs.jsonl
input_filename = '/home/workspace/lgq/shop/data/query_instruction_pairs.jsonl' 

# 您可以按需修改输出文件的名称
train_output_filename = '/home/workspace/lgq/shop/data/query_instruction_pairs_train_set.jsonl'
test_output_filename = '/home/workspace/lgq/shop/data/query_instruction_pairs_test_set.jsonl'
# --- 配置结束 ---


def split_jsonl_dataset(input_file, train_file, test_file):
    """
    读取一个 JSONL 格式的文件，并将其按奇偶行拆分为训练集和测试集。

    Args:
        input_file (str): 输入的源文件名。
        train_file (str): 输出的训练集文件名。
        test_file (str): 输出的测试集文件名。
    """
    print(f"开始处理文件: '{input_file}'")
    
    try:
        train_count = 0
        test_count = 0
        
        # 使用 with open(...) as ... 语法可以确保文件在处理后被正确关闭
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(train_file, 'w', encoding='utf-8') as train_outfile, \
             open(test_file, 'w', encoding='utf-8') as test_outfile:

            # 使用 enumerate 来获取行号，行号从 1 开始
            for i, line in enumerate(infile, 1):
                # 检查行是否为空白行，如果是则跳过
                if not line.strip():
                    continue

                # 判断行号是奇数还是偶数
                if i % 2 != 0:
                    # 奇数行 (1, 3, 5, ...)，写入训练集
                    train_outfile.write(line)
                    train_count += 1
                else:
                    # 偶数行 (2, 4, 6, ...)，写入测试集
                    test_outfile.write(line)
                    test_count += 1
        
        print("-" * 30)
        print("数据集拆分成功！")
        print(f"共写入 {train_count} 条数据到训练集: '{train_file}'")
        print(f"共写入 {test_count} 条数据到测试集: '{test_file}'")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        print("请确保文件名正确，并且文件与脚本在同一目录下。")
    except Exception as e:
        print(f"处理过程中发生了一个意料之外的错误: {e}")


# --- 脚本执行 ---
if __name__ == '__main__':
    # 在运行前，先检查输入文件是否存在
    if os.path.exists(input_filename):
        split_jsonl_dataset(input_filename, train_output_filename, test_output_filename)
    else:
        # 如果文件不存在，给出清晰的提示
        print(f"错误: 无法开始处理，因为输入文件 '{input_filename}' 不存在。")
        print("请将您的数据文件放在与脚本相同的目录下，并更新脚本中的 'input_filename' 变量。")