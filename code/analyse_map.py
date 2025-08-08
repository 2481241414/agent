import pandas as pd
from datetime import datetime
import ast
import os

def dict_list_to_set(lst):
    """
    将列表如 [{"快速": "曹操出行"}, {"推荐": "特快出租车"}] 转为 set([frozenset({...}), ...])
    方便集合运算
    """
    if not isinstance(lst, list):
        return set()
    return set([frozenset(d.items()) for d in lst if isinstance(d, dict)])

def get_prec_and_rec(ans, ground):
    # ans, ground 为字符串形式的列表
    try:
        ans_list = ast.literal_eval(ans) if ans and ans != 'nan' else []
        ground_list = ast.literal_eval(ground) if ground and ground != 'nan' else []
    except Exception:
        ans_list, ground_list = [], []
    ans_set = dict_list_to_set(ans_list)
    ground_set = dict_list_to_set(ground_list)
    intersect = len(ans_set & ground_set)
    # 完全都为空，视为完全正确
    if len(ans_set) == 0 and len(ground_set) == 0:
        return {'prec': 1, 'rec': 1, 'f1': 1, 'proper': 1}
    # 只要有一方为空都视为0分
    if len(ans_set) == 0 and len(ground_set) != 0:
        return {'prec': 0, 'rec': 0, 'f1': 0, 'proper': 0}
    if len(ans_set) != 0 and len(ground_set) == 0:
        return {'prec': 0, 'rec': 0, 'f1': 0, 'proper': 0}
    # 有误选（多选），精确率直接为0
    if len(ans_set - ground_set) > 0:
        return {'prec': 0, 'rec': intersect / len(ground_set), 'f1': 0, 'proper': 0}
    # 没有误选，精确率为1，否则为0
    prec = 1 if ans_set == ground_set or ans_set <= ground_set else 0
    rec = intersect / len(ground_set)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    proper = 1 if prec == 1 and rec == 1 else 0
    return {'prec': prec, 'rec': rec, 'f1': f1, 'proper': proper}

def analyse(file_path: str, version='v1.0'):
    df = pd.read_csv(file_path)
    metric = []
    for index, row in df.iterrows():
        metric.append(
            get_prec_and_rec(row['模型输出车型组合'], row['正确车型组合'])
        )
    df_metric = pd.DataFrame(metric)
    final_output = pd.concat([df, df_metric], axis=1)

    prec_mean = df_metric['prec'].mean()
    rec_mean = df_metric['rec'].mean()
    f1_mean = df_metric['f1'].mean()
    proper_rate = df_metric['proper'].mean()

    print(f'**精确率（Precision）：{prec_mean:.4f}**')
    print(f'**召回率（Recall）：{rec_mean:.4f}**')
    print(f'**F1：{f1_mean:.4f}**')
    print(f'**完全正确比例（proper）：{proper_rate:.4f}**')

    # 日期（精确到日）
    date_str = datetime.now().strftime("%Y%m%d")
    # 构建输出路径
    output_dir = f'/home/workspace/lgq/data/inference/{date_str}/{version}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'analyse_output.csv')
    
    final_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'**文件已保存至：{output_path}**')

if __name__ == '__main__':
    date_str = datetime.now().strftime("%m%d")
    # analyse(file_path='/home/workspace/lgq/data/inference/merge_data_inference_results.csv', version='v1.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/merge_data_inference_results_{date_str}.csv', version='v1.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/merge_data_map_v3.0_FINAL_inference_results_{date_str}.csv', version='v2.0')
    #analyse(file_path=f'/home/workspace/lgq/data/inference/test_dataset_05_inference_results_{date_str}.csv', version='v2.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/20250619/v2.0/test_dataset_05_inference_results_0619.csv', version='v2.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/20250620/v1.0/打车语料2.0(多轮query)_sft_inference_results_0620.csv', version='1.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/20250625/v1.0/test_dataset_05_inference_results_0625.csv', version='v1.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/20250625/v2.0/打车语料-2.0版-624-zhao - 车型理解_inference_results_0625.csv', version='v2.0')
    analyse(file_path=f'/home/workspace/lgq/data/inference/20250625/v2.0/test_dataset_05_inference_results_0625.csv', version='v2.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/20250702/v1.0/test_dataset_05_inference_results_0702.csv', version='v1.0')
    # analyse(file_path=f'/home/workspace/lgq/data/inference/20250702/v1.0/打车语料-2.0版-624-zhao - 车型理解_inference_results_0702.csv', version='v1.0')


