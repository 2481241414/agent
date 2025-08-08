import pandas as pd
from datetime import datetime

def get_prec_and_rec(ans, ground):
    ans, ground = eval(ans), eval(ground)
    intersect = len(set(ground) & set(ans))
    if len(ans) == 0 and len(ground) == 0:
        return {'prec': 1, 'rec': 1, 'f1': 1, 'proper': 1}
    if len(ans) == 0 and len(ground) != 0:
        return {'prec': 0, 'rec': 0, 'f1': 0, 'proper': 0}
    if len(ans) != 0 and len(ground) == 0:
        return {'prec': 0, 'rec': 0, 'f1': 0, 'proper': 0}
    # 误召回算0分
    if len(ans) - len(ground) > 0:
        return {'prec': 0, 'rec': intersect / len(ans), 'f1': 0, 'proper': 0}
    prec = intersect / len(ans)
    rec = intersect / len(ground)
    if prec == 1 and rec == 1:
        return {
            'prec': prec,
            'rec': rec,
            'f1': 2 * (intersect) / (len(ans) + len(ground)),
            'proper': 1
        }
    else:
        return {
            'prec': prec,
            'rec': rec,
            'f1': 2 * (intersect) / (len(ans) + len(ground)),
            'proper': 0
        }

def analyse(file_path: str):
    df = pd.read_csv(file_path)
    metric = []
    for index, row in df.iterrows():
        metric.append(
            get_prec_and_rec(row['车型大类'], row['大类正确答案'])
        )
    df_category = pd.DataFrame(metric).rename(columns={'proper': '大类proper'})

    final_output = pd.concat([df, df_category['大类proper']], axis=1)

    prec_mean = df_category['prec'].mean()
    rec_mean = df_category['rec'].mean()
    print(f'大类准确率：{prec_mean}')
    print(f'大类召回率：{rec_mean}')

    metric.clear()
    for index, row in df.iterrows():
        metric.append(
            get_prec_and_rec(row['车辆类型'], row['小类正确答案'])
        )
    df_subcategory = pd.DataFrame(metric).rename(columns={'proper': '小类proper'})
    final_output = pd.concat([final_output, df_subcategory['小类proper']], axis=1)

    prec_mean = df_subcategory['prec'].mean()
    rec_mean = df_subcategory['rec'].mean()
    print(f'小类准确率：{prec_mean}')
    print(f'小类召回率：{rec_mean}')

    # 生成精确到秒的时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'/home/workspace/lgq/data/inference/analyse_output_{timestamp}.xlsx'
    final_output.to_excel(output_path, index=False)
    print(f'文件已保存至：{output_path}')

if __name__ == '__main__':
    analyse(file_path='/home/workspace/lgq/data/inference/merge_results_3.csv')
