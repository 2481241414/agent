import chardet
import pandas as pd
import re
import csv

###########################################把#$格式的输出转换为csv格式
file_path = '500指令数据集-5-标注-process.csv'  # 替换为实际的文件路径
df = pd.read_csv(file_path, encoding='gbk')

batch_size = 10  # 每10行保存一次

for index, row in df.iterrows():
    text = row['输出']
    # 使用正则表达式提取所需信息
    match = re.search(r'<Plan>#(.*?)#.*\$(.*?)\$.*</Plan><Finish>', text)
    if match:
        category = match.group(1)
        app_name = match.group(2)
    else:
        print(f"第{index}行未找到匹配的模式")
        continue
    plan_content_match = re.search(r'<Plan>(.*?)</Plan>', text)
    if plan_content_match:
        plan_content = plan_content_match.group(1)
        task = re.sub(r'#.*?#', '', plan_content)
        task = re.sub(r'\$(.*?)\$', r'\1', task)
        task = task.strip()
    else:
        print(f"第{index}行未找到Plan内容")
        continue

    cell_content = f"taskId,task,category,app_name\n1,{task},{category},{app_name}"
    print("原始输出为：",  text)
    print(f"第{index}行处理完成，输出为：", cell_content)
    df.at[index, 'csv_output'] = cell_content

    # 每处理完 batch_size 行，保存一次
    if (index + 1) % batch_size == 0:
        df.to_csv('500指令数据集-5-标注-process.csv', index=False, encoding='gbk')
        print(f"已保存第 {index + 1} 行数据")
    
# 处理完所有行后，保存剩余未保存的数据
df.to_csv('500指令数据集-5-标注-process.csv', index=False, encoding='gbk')
print("全部数据处理完成并保存")