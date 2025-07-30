import csv
import os

# --- 配置 ---
# 请将 'input.csv' 替换为您的源文件名
input_filename = '/home/workspace/lgq/shop/data/single_gt_output_with_plan.csv' 

# 您可以按需修改输出文件的名称
output_filename = '/home/workspace/lgq/shop/data/single_gt_output_with_plan_evel_rows_dataset.csv'
# --- 配置结束 ---


def extract_even_data_rows(input_file, output_file):
    """
    读取一个 CSV 文件，并将表头和所有偶数行数据写入一个新的CSV文件。

    Args:
        input_file (str): 输入的源CSV文件名。
        output_file (str): 输出的目标CSV文件名。
    """
    print(f"开始处理文件: '{input_file}'")
    
    try:
        # 使用 'utf-8-sig' 来正确处理可能由Excel等软件保存的带BOM的UTF-8文件
        # newline='' 是写入CSV文件时的标准做法，防止出现空行
        with open(input_file, 'r', encoding='utf-8-sig') as infile, \
             open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 1. 读取并写入表头
            # next(reader)会读取当前行并移动到下一行
            header = next(reader)
            writer.writerow(header)

            # 2. 遍历数据行，提取偶数行
            # enumerate从0开始计数，所以我们找索引为奇数的行（1, 3, 5...），它们对应实际的第2, 4, 6行数据
            even_rows_count = 0
            for i, row in enumerate(reader):
                if (i + 1) % 2 == 0:
                    # i从0开始，所以第2行数据的i是1，第4行是3，以此类推。
                    # (i + 1) % 2 == 0 完美匹配偶数行。
                    writer.writerow(row)
                    even_rows_count += 1
        
        print("-" * 30)
        print("处理成功！")
        print(f"共提取了 {even_rows_count} 条偶数行数据。")
        print(f"结果已保存至: '{output_file}'")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        print("请确保文件名正确，并且文件与脚本在同一目录下。")
    except StopIteration:
        # 这个错误会在文件为空或只有表头时发生
        print("警告: 输入文件为空或只包含表头，没有数据可以提取。")
    except Exception as e:
        print(f"处理过程中发生了一个意料之外的错误: {e}")


# --- 脚本执行 ---
if __name__ == '__main__':



    # 运行提取函数
    extract_even_data_rows(input_filename, output_filename)