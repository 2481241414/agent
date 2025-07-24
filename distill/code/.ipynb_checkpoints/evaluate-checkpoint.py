# run_evaluation.py (强烈建议您重命名此文件)

import os
import json
import csv
import re
from collections import Counter # 导入 Counter 用于高效计数
import math # 导入 math 用于计算对数和指数

# --- 1. 文件路径配置 ---
# INPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250714/inference_results.json"
# OUTPUT_EVALUATION_PATH = "/home/workspace/lgq/distill/data/20250714/evaluation_report.csv"

# INPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250715/7b_inference_results_async.json"
# INPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250715/7b_inference_results_async_2query.json"
# INPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250716/7b_inference_results_async_2query.json"
INPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250717/Qwen2.5-72b_results_async_2query_0717.json"

# OUTPUT_EVALUATION_PATH = "/home/workspace/lgq/distill/data/20250715/7b_evaluation_report.csv"
# OUTPUT_EVALUATION_PATH = "/home/workspace/lgq/distill/data/20250715/7b_evaluation_report_2query.csv"
# OUTPUT_EVALUATION_PATH = "/home/workspace/lgq/distill/data/20250716/7b_evaluation_report_2query.csv"
OUTPUT_EVALUATION_PATH = "/home/workspace/lgq/distill/data/20250717/Qwen2.5-72B_evaluation_report_2query_0717.csv"



def _get_ngrams(tokens, n):
    """从 token 列表中提取 n-grams"""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def calculate_bleu(prediction_str, reference_str):
    """
    手动计算 BLEU-1, 2, 3, 4 的精度分量。
    使用基于字符的分词，对中文和结构化文本更友好。
    """
    # 修改点：将 split() 改为 list()，实现按字符分词
    prediction_tokens = list(prediction_str)
    reference_tokens = list(reference_str)
    
    len_pred = len(prediction_tokens)
    if len_pred == 0:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

    scores = {}
    for n in range(1, 5):
        if len_pred < n:
            scores[f'bleu_{n}'] = 0.0
            continue
            
        pred_ngrams = Counter(_get_ngrams(prediction_tokens, n))
        ref_ngrams = Counter(_get_ngrams(reference_tokens, n))
        
        clipped_count = sum((pred_ngrams & ref_ngrams).values())
        
        total_pred_ngrams = max(1, len_pred - n + 1)
        scores[f'bleu_{n}'] = clipped_count / total_pred_ngrams

    # 注意：一个完整的BLEU实现还包括简洁度惩罚(Brevity Penalty)。
    # 为保持简单，这里返回的是n-gram精度，这在很多场景下已足够用于比较。
    return scores

def _lcs_length(a, b):
    """计算两个列表的最长公共子序列长度 (LCS)"""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]

def calculate_rouge(prediction_str, reference_str):
    """
    手动计算 ROUGE-1, ROUGE-2, ROUGE-L 分数。
    使用基于字符的分词。
    """
    # 修改点：将 split() 改为 list()，实现按字符分词
    prediction_tokens = list(prediction_str)
    reference_tokens = list(reference_str)
    
    len_pred = len(prediction_tokens)
    len_ref = len(reference_tokens)

    if len_pred == 0 or len_ref == 0:
        return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}

    scores = {}
    # ROUGE-N (基于F1分数，这是更标准的做法)
    for n in [1, 2]:
        pred_ngrams = set(_get_ngrams(prediction_tokens, n))
        ref_ngrams = set(_get_ngrams(reference_tokens, n))
        
        match = len(pred_ngrams.intersection(ref_ngrams))
        recall = match / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0
        precision = match / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        scores[f'rouge_{n}'] = f1
    
    # ROUGE-L (基于F1分数)
    lcs_len = _lcs_length(prediction_tokens, reference_tokens)
    recall = lcs_len / len_ref if len_ref > 0 else 0.0
    precision = lcs_len / len_pred if len_pred > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    scores['rouge_l'] = f1
    
    return scores

def clean_text(text):
    """清理文本，去除前后空白。"""
    return text.strip()

def extract_info(text: str):
    """
    使用正则表达式从模型输出中提取大类和应用名。
    """
    categories = re.findall(r'#([^#]+)#', text)
    app_names = re.findall(r'\$([^$]+)\$', text)
    
    return categories, app_names

def main():
    """主评测流程"""
    try:
        with open(INPUT_INFERENCE_PATH, 'r', encoding='utf-8') as f:
            inference_results = json.load(f)
        print(f"成功加载推理结果文件，共 {len(inference_results)} 条。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 加载文件 '{INPUT_INFERENCE_PATH}' 失败 - {e}")
        return

    # CSV 表头与之前版本保持一致
    csv_headers = [
        'user_input', 'ground_truth_output', 'model_output', 'overall_correct',
        'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
        'rouge_1', 'rouge_2', 'rouge_l',
        'gt_categories', 'model_categories', 'category_correct',
        'gt_app_names', 'model_app_names', 'app_name_correct', 'delay_seconds'
    ]
    with open(OUTPUT_EVALUATION_PATH, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)
    
    # 初始化字典来存放所有指标的总和
    total_count = 0
    overall_correct_count = 0
    category_correct_count = 0
    app_name_correct_count = 0
    total_scores = {
        'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
        'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0,
    }
    
    print("开始进行评测 (使用内建函数)...")
    # 主评测循环
    for item in inference_results:
        user_input = item.get("input", "")
        ground_truth = item.get("ground_truth", "")
        model_output = item.get("model_output", "")
        delay = item.get("delay_seconds", -1.0)
        
        is_overall_correct, is_category_correct, is_app_name_correct = 0, 0, 0
        current_scores = {key: 0.0 for key in total_scores}

        gt_categories, gt_app_names = extract_info(ground_truth)
        model_categories, model_app_names = extract_info(model_output)
        
        cleaned_model_output = clean_text(model_output)
        cleaned_ground_truth = clean_text(ground_truth)

        if "ERROR:" in model_output or not cleaned_model_output:
            pass # 所有指标均为0
        else:
            # 修改: 调用我们自己写的函数
            bleu_result = calculate_bleu(cleaned_model_output, cleaned_ground_truth)
            rouge_result = calculate_rouge(cleaned_model_output, cleaned_ground_truth)
            
            # 合并结果
            current_scores.update(bleu_result)
            current_scores.update(rouge_result)
            
            if cleaned_model_output == cleaned_ground_truth:
                is_overall_correct, is_category_correct, is_app_name_correct = 1, 1, 1
                for key in current_scores: current_scores[key] = 1.0
            else:
                if sorted(gt_categories) == sorted(model_categories): is_category_correct = 1
                if sorted(gt_app_names) == sorted(model_app_names): is_app_name_correct = 1
        
        # 更新总计数器
        total_count += 1
        overall_correct_count += is_overall_correct
        category_correct_count += is_category_correct
        app_name_correct_count += is_app_name_correct
        for key in total_scores:
            total_scores[key] += current_scores[key]

        # 写入 CSV
        with open(OUTPUT_EVALUATION_PATH, 'a', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                user_input, ground_truth, model_output, is_overall_correct,
                f"{current_scores['bleu_1']:.4f}", f"{current_scores['bleu_2']:.4f}",
                f"{current_scores['bleu_3']:.4f}", f"{current_scores['bleu_4']:.4f}",
                f"{current_scores['rouge_1']:.4f}", f"{current_scores['rouge_2']:.4f}",
                f"{current_scores['rouge_l']:.4f}",
                str(gt_categories), str(model_categories), is_category_correct,
                str(gt_app_names), str(model_app_names), is_app_name_correct,
                f"{delay:.4f}"
            ])
            
    # 输出最终评测报告
    if total_count > 0:
        overall_accuracy = (overall_correct_count / total_count) * 100
        category_accuracy = (category_correct_count / total_count) * 100
        app_name_accuracy = (app_name_correct_count / total_count) * 100
        
        average_scores = {key: val / total_count for key, val in total_scores.items()}
        
        print("\n--- 评测完成 ---")
        print(f"总计评测数据条数: {total_count}")
        print("-" * 30)
        print("--- 整体与内容质量评估 ---")
        print(f"完全匹配准确率 (Exact Match): {overall_accuracy:.2f}% ({overall_correct_count}/{total_count})")
        print(f"Avg. BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4: "
              f"{average_scores['bleu_1']:.4f} / "
              f"{average_scores['bleu_2']:.4f} / "
              f"{average_scores['bleu_3']:.4f} / "
              f"{average_scores['bleu_4']:.4f}")
        print(f"Avg. ROUGE-1 / ROUGE-2 / ROUGE-L: "
              f"{average_scores['rouge_1']:.4f} / "
              f"{average_scores['rouge_2']:.4f} / "
              f"{average_scores['rouge_l']:.4f}")
        print("-" * 30)
        print("--- 特定信息提取评估 ---")
        print(f"大类准确率 (Category Accuracy): {category_accuracy:.2f}% ({category_correct_count}/{total_count})")
        print(f"应用名准确率 (App Name Accuracy): {app_name_accuracy:.2f}% ({app_name_correct_count}/{total_count})")
        print("-" * 30)
        print(f"详细评测报告已保存到: {os.path.abspath(OUTPUT_EVALUATION_PATH)}")
    else:
        print("没有可评测的数据。")

if __name__ == "__main__":
    main()