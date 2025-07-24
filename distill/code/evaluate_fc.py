# run_evaluation.py (Added 'tag' column)

import json
import re
import ast
import csv
import os
from collections import Counter
import jieba
from thefuzz import fuzz

# --- 文件路径配置 ---
INPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250721/fc/qwen2.5_72b_int8_FC能力验证_inference_results.json"
OUTPUT_EVALUATION_DETAILS_PATH = "/home/workspace/lgq/distill/data/20250721/fc/qwen2.5_72b_int8_FC能力验证_evaluation.csv"


# --- 1. 核心辅助函数 ---
def parse_dsl_string(dsl_str: str):
    if not dsl_str or not dsl_str.strip(): return None, {}
    try:
        dsl_list = ast.literal_eval(dsl_str)
        if not isinstance(dsl_list, list) or not dsl_list or len(dsl_list[0]) < 2: return None, {}
        call_str = dsl_list[0][1]
        match = re.match(r'(\w+)\((.*)\)', call_str)
        if not match:
            no_param_match = re.match(r'(\w+)\(?\)?', call_str)
            if no_param_match: return no_param_match.group(1), {}
            return None, {}
        tool_name, params_str = match.group(1), match.group(2)
        params = dict(re.findall(r'(\w+)\s*=\s*"(.*?)"', params_str))
        return tool_name, params
    except (ValueError, SyntaxError, IndexError): return None, {}

def parse_tools_from_prompt(user_prompt: str):
    tools_def = {}
    try:
        tools_match = re.search(r'<tools>(.*?)</tools>', user_prompt, re.DOTALL)
        if not tools_match: return {}
        tools_str = tools_match.group(1).strip()
        json_str = f"[{tools_str.replace('}{', '},{')}]"
        tool_list = json.loads(json_str)
        for tool in tool_list:
            if tool.get('function'):
                func = tool['function']
                tool_name, properties = func.get('name'), func.get('parameters', {}).get('properties', {})
                if tool_name: tools_def[tool_name] = properties
    except (json.JSONDecodeError, AttributeError): return {}
    return tools_def

def is_open_domain(param_name: str, tool_name: str, tools_def: dict):
    if param_name == 'app': return False
    if tool_name in tools_def and param_name in tools_def[tool_name]:
        return 'enum' not in tools_def[tool_name][param_name]
    return True

def safe_div(num, den, to_percent=True):
    if den == 0: return 0.0
    result = num / den
    return result * 100 if to_percent else result

def calculate_prf1(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def _get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def calculate_bleu(prediction_str, reference_str):
    prediction_tokens = jieba.lcut(prediction_str.strip())
    reference_tokens = jieba.lcut(reference_str.strip())
    len_pred = len(prediction_tokens)
    if len_pred == 0: return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    scores = {}
    for n in range(1, 5):
        if len_pred < n: scores[f'bleu_{n}'] = 0.0; continue
        pred_ngrams, ref_ngrams = Counter(_get_ngrams(prediction_tokens, n)), Counter(_get_ngrams(reference_tokens, n))
        clipped_count = sum((pred_ngrams & ref_ngrams).values())
        total_pred_ngrams = max(1, len_pred - n + 1)
        scores[f'bleu_{n}'] = clipped_count / total_pred_ngrams
    return scores

def _lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]: dp[i][j] = dp[i - 1][j - 1] + 1
            else: dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]

def calculate_rouge(prediction_str, reference_str):
    prediction_tokens, reference_tokens = jieba.lcut(prediction_str.strip()), jieba.lcut(reference_str.strip())
    len_pred, len_ref = len(prediction_tokens), len(reference_tokens)
    if len_pred == 0 or len_ref == 0: return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
    scores = {}
    for n in [1, 2]:
        pred_ngrams, ref_ngrams = set(_get_ngrams(prediction_tokens, n)), set(_get_ngrams(reference_tokens, n))
        match = len(pred_ngrams.intersection(ref_ngrams))
        recall, precision = match / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0, match / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        scores[f'rouge_{n}'] = f1
    lcs_len = _lcs_length(prediction_tokens, reference_tokens)
    recall_l, precision_l = lcs_len / len_ref if len_ref > 0 else 0.0, lcs_len / len_pred if len_pred > 0 else 0.0
    f1_l = 2 * (precision_l * recall_l) / (precision_l + recall_l) if (precision_l + recall_l) > 0 else 0.0
    scores['rouge_l'] = f1_l
    return scores

# --- 2. 评估主函数 ---

def run_evaluation_and_get_details(inference_results):
    """对每个样本进行评估，并返回包含逐行评估细节的列表。"""
    detailed_results = []
    
    for result in inference_results:
        # --- 新增: 加入 '标签(tag)' ---
        eval_row = {
            '标签(tag)': result.get('tag', ''),
            '用户输入': result.get('user_prompt'),
            '标准答案': result.get('ground_truth'),
            '模型输出': result.get('model_output'),
            '时延(秒)': result.get('delay_seconds', -1.0),
            '格式正确率': 0, '综合准确率(函数名+参数名+闭域值)': 0, '完整匹配准确率': 0,
            '函数名准确率': 0, '参数结构准确率': 0,
            '参数名-精确率': 0.0, '参数名-召回率': 0.0, '参数名-F1': 0.0,
            '闭域参数值-准确率': None, '开域参数值-Fuzz相似度': None,
            '开域参数值-BLEU1': None, '开域参数值-BLEU2': None, '开域参数值-BLEU3': None, '开域参数值-BLEU4': None,
            '开域参数值-ROUGE1': None, '开域参数值-ROUGE2': None, '开域参数值-ROUGEL': None,
        }
        
        tools_def = parse_tools_from_prompt(result.get("user_prompt", ""))
        gt_tool, gt_params = parse_dsl_string(result.get("ground_truth", ""))
        pred_tool, pred_params = parse_dsl_string(result.get("model_output", ""))

        if pred_tool is not None: eval_row['格式正确率'] = 1
        if not gt_tool:
            detailed_results.append(eval_row); continue
            
        if result.get("ground_truth") == result.get("model_output"): eval_row['完整匹配准确率'] = 1
        
        if gt_tool == pred_tool:
            eval_row['函数名准确率'] = 1
            gt_keys, pred_keys = set(gt_params.keys()), set(pred_params.keys())
            
            sample_tp, sample_fp, sample_fn = len(gt_keys & pred_keys), len(pred_keys - gt_keys), len(gt_keys - pred_keys)
            precision, recall, f1 = calculate_prf1(sample_tp, sample_fp, sample_fn)
            eval_row['参数名-精确率'], eval_row['参数名-召回率'], eval_row['参数名-F1'] = precision, recall, f1
            
            if gt_keys == pred_keys:
                eval_row['参数结构准确率'] = 1
                all_closed_values_match = True
                for key in gt_keys:
                    if not is_open_domain(key, gt_tool, tools_def) and gt_params.get(key) != pred_params.get(key):
                        all_closed_values_match = False; break
                if all_closed_values_match: eval_row['综合准确率(函数名+参数名+闭域值)'] = 1
            
            closed_correct, closed_total = 0, 0
            open_fuzz, open_bleu, open_rouge = [], [], []
            for key in (gt_keys & pred_keys):
                if is_open_domain(key, gt_tool, tools_def):
                    pred_val, gt_val = pred_params[key], gt_params[key]
                    open_fuzz.append(fuzz.ratio(pred_val, gt_val))
                    if pred_val and gt_val:
                        bleu_scores, rouge_scores = calculate_bleu(pred_val, gt_val), calculate_rouge(pred_val, gt_val)
                        open_bleu.append(bleu_scores); open_rouge.append(rouge_scores)
                else:
                    closed_total += 1
                    if gt_params[key] == pred_params[key]: closed_correct += 1
            
            if closed_total > 0: eval_row['闭域参数值-准确率'] = safe_div(closed_correct, closed_total)
            if open_fuzz: eval_row['开域参数值-Fuzz相似度'] = sum(open_fuzz) / len(open_fuzz)
            if open_bleu:
                eval_row['开域参数值-BLEU1'] = safe_div(sum(b['bleu_1'] for b in open_bleu), len(open_bleu), False)
                eval_row['开域参数值-BLEU2'] = safe_div(sum(b['bleu_2'] for b in open_bleu), len(open_bleu), False)
                eval_row['开域参数值-BLEU3'] = safe_div(sum(b['bleu_3'] for b in open_bleu), len(open_bleu), False)
                eval_row['开域参数值-BLEU4'] = safe_div(sum(b['bleu_4'] for b in open_bleu), len(open_bleu), False)
            if open_rouge:
                eval_row['开域参数值-ROUGE1'] = safe_div(sum(r['rouge_1'] for r in open_rouge), len(open_rouge), False)
                eval_row['开域参数值-ROUGE2'] = safe_div(sum(r['rouge_2'] for r in open_rouge), len(open_rouge), False)
                eval_row['开域参数值-ROUGEL'] = safe_div(sum(r['rouge_l'] for r in open_rouge), len(open_rouge), False)
        
        detailed_results.append(eval_row)
        
    return detailed_results

def save_details_to_csv(detailed_results, filepath):
    if not detailed_results: return
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    headers = list(detailed_results[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(detailed_results)
    print(f"\n详细评估结果已保存到: {filepath}")

def print_summary_report(detailed_results):
    total_samples = len(detailed_results)
    if total_samples == 0: return

    accuracy_format = safe_div(sum(r['格式正确率'] for r in detailed_results), total_samples)
    accuracy_combined = safe_div(sum(r['综合准确率(函数名+参数名+闭域值)'] for r in detailed_results), total_samples)
    accuracy_dsl = safe_div(sum(r['完整匹配准确率'] for r in detailed_results), total_samples)
    func_correct_count = sum(r['函数名准确率'] for r in detailed_results)
    accuracy_func = safe_div(func_correct_count, total_samples)
    accuracy_pname_struct = safe_div(sum(r['参数结构准确率'] for r in detailed_results), func_correct_count)

    valid_prf_samples = [r for r in detailed_results if r['函数名准确率']]
    avg_precision_pname = sum(r['参数名-精确率'] for r in valid_prf_samples) / len(valid_prf_samples) if valid_prf_samples else 0
    avg_recall_pname = sum(r['参数名-召回率'] for r in valid_prf_samples) / len(valid_prf_samples) if valid_prf_samples else 0
    avg_f1_pname = sum(r['参数名-F1'] for r in valid_prf_samples) / len(valid_prf_samples) if valid_prf_samples else 0

    closed_val_samples = [r for r in detailed_results if r['闭域参数值-准确率'] is not None]
    avg_cval_accuracy = sum(r['闭域参数值-准确率'] for r in closed_val_samples) / len(closed_val_samples) if closed_val_samples else 0
    
    open_domain_samples = [r for r in detailed_results if r['开域参数值-Fuzz相似度'] is not None]
    avg_fuzz = sum(r['开域参数值-Fuzz相似度'] for r in open_domain_samples) / len(open_domain_samples) if open_domain_samples else 0
    avg_bleu1 = safe_div(sum(r['开域参数值-BLEU1'] for r in open_domain_samples if r['开域参数值-BLEU1'] is not None), len(open_domain_samples))
    avg_bleu2 = safe_div(sum(r['开域参数值-BLEU2'] for r in open_domain_samples if r['开域参数值-BLEU2'] is not None), len(open_domain_samples))
    avg_bleu3 = safe_div(sum(r['开域参数值-BLEU3'] for r in open_domain_samples if r['开域参数值-BLEU3'] is not None), len(open_domain_samples))
    avg_bleu4 = safe_div(sum(r['开域参数值-BLEU4'] for r in open_domain_samples if r['开域参数值-BLEU4'] is not None), len(open_domain_samples))
    avg_rouge1 = safe_div(sum(r['开域参数值-ROUGE1'] for r in open_domain_samples if r['开域参数值-ROUGE1'] is not None), len(open_domain_samples))
    avg_rouge2 = safe_div(sum(r['开域参数值-ROUGE2'] for r in open_domain_samples if r['开域参数值-ROUGE2'] is not None), len(open_domain_samples))
    avg_rougeL = safe_div(sum(r['开域参数值-ROUGEL'] for r in open_domain_samples if r['开域参数值-ROUGEL'] is not None), len(open_domain_samples))
    
    print("\n" + "="*20 + " 模型综合评估报告 " + "="*20)
    print(f"总样本数: {total_samples}")
    print("\n--- [ 基础评估: 格式与匹配 ] ---")
    print(f"格式正确率: {accuracy_format:.2f}%")
    print(f"综合准确率 (函数名+参数结构+闭域值): {accuracy_combined:.2f}%")
    print(f"完整匹配准确率 (Full Match):    {accuracy_dsl:.2f}%")
    print("\n--- [ 级别 1: 函数名评估 ] ---")
    print(f"准确率: {accuracy_func:.2f}%  (正确: {func_correct_count})")
    print("\n--- [ 级别 2: 参数名评估 (当函数名正确时) ] ---")
    print(f"参数结构准确率: {accuracy_pname_struct:.2f}%")
    print(f"宏观平均-精确率: {avg_precision_pname:.2f}%")
    print(f"宏观平均-召回率: {avg_recall_pname:.2f}%")
    print(f"宏观平均-F1:     {avg_f1_pname:.2f}%")
    print("\n--- [ 级别 3: 参数值评估 (当参数名正确时) ] ---")
    print("\n  -- 闭域参数值 --")
    print(f"准确率: {avg_cval_accuracy:.2f}%")
    print("\n  -- 开域参数值 --")
    print(f"Avg. Fuzz相似度: {avg_fuzz:.2f}")
    print(f"Avg. BLEU-1/2/3/4: {avg_bleu1:.2f} / {avg_bleu2:.2f} / {avg_bleu3:.2f} / {avg_bleu4:.2f}")
    print(f"Avg. ROUGE-1/2/L:  {avg_rouge1:.2f} / {avg_rouge2:.2f} / {avg_rougeL:.2f}")
    print(f"(评估样本数: {len(open_domain_samples)})")
    print("\n" + "="*58)

def main():
    try:
        with open(INPUT_INFERENCE_PATH, 'r', encoding='utf-8') as f:
            inference_results = json.load(f)
        print(f"成功加载 {len(inference_results)} 条推理结果用于评估。")
    except Exception as e:
        print(f"错误: 加载推理结果文件 '{INPUT_INFERENCE_PATH}' 失败 - {e}")
        return
    
    detailed_results = run_evaluation_and_get_details(inference_results)
    save_details_to_csv(detailed_results, OUTPUT_EVALUATION_DETAILS_PATH)
    print_summary_report(detailed_results)

if __name__ == "__main__":
    main()