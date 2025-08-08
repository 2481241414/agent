# run_evaluation.py (Grouped by Tag)

import json
import re
import ast
import csv
import os
from collections import Counter
import jieba
from thefuzz import fuzz

# --- 文件路径配置 ---
INPUT_INFERENCE_PATH = "/home/workspace/lgq/distill/data/20250723/fc/yunke_FC能力验证_inference_results.json"
OUTPUT_EVALUATION_DETAILS_PATH = "/home/workspace/lgq/distill/data/20250723/fc/Qwen2.5-7B-yunke_FC能力验证_evaluation_tag.csv"


# --- 1. 核心辅助函数 ---
def parse_dsl_string(dsl_str: str):
    """解析DSL字符串，返回(tool_name, params_dict)"""
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
    """从Prompt中解析工具定义"""
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
    """判断参数是否为开域参数"""
    if param_name == 'app': return False
    if tool_name in tools_def and param_name in tools_def[tool_name]:
        return 'enum' not in tools_def[tool_name][param_name]
    return True

def safe_div(num, den, to_percent=True):
    """安全除法"""
    if den == 0: return 0.0
    result = num / den
    return result * 100 if to_percent else result

def calculate_prf1(tp, fp, fn):
    """计算精确率、召回率和F1值"""
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def _get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def calculate_bleu(prediction_str, reference_str):
    prediction_tokens, reference_tokens = jieba.lcut(prediction_str.strip()), jieba.lcut(reference_str.strip())
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
        recall = match / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0
        precision = match / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        scores[f'rouge_{n}'] = f1
    lcs_len = _lcs_length(prediction_tokens, reference_tokens)
    recall_l = lcs_len / len_ref if len_ref > 0 else 0.0
    precision_l = lcs_len / len_pred if len_pred > 0 else 0.0
    f1_l = 2 * (precision_l * recall_l) / (precision_l + recall_l) if (precision_l + recall_l) > 0 else 0.0
    scores['rouge_l'] = f1_l
    return scores


# --- 2. 评估主函数 ---
def run_evaluation_and_get_details(inference_results):
    detailed_results = []
    for result in inference_results:
        eval_row = {
            '标签(tag)': result.get('tag', ''), '用户输入': result.get('user_prompt'),
            '标准答案': result.get('ground_truth'), '模型输出': result.get('model_output'),
            '时延(秒)': result.get('delay_seconds', -1.0),
            '格式正确率': 0,
            '宽松综合准确率(必需参数)': 0,
            '严格综合准确率(全参数)': 0,
            '完整匹配准确率': 0,
            '函数名准确率': 0,
            '参数结构准确率(宽松)': 0,
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
            detailed_results.append(eval_row)
            continue
            
        if result.get("ground_truth") == result.get("model_output"):
            eval_row['完整匹配准确率'] = 1

        if gt_tool == pred_tool:
            eval_row['函数名准确率'] = 1
            gt_keys, pred_keys = set(gt_params.keys()), set(pred_params.keys())
            
            sample_tp = len(gt_keys & pred_keys)
            sample_fp = len(pred_keys - gt_keys)
            sample_fn = len(gt_keys - pred_keys)
            precision, recall, f1 = calculate_prf1(sample_tp, sample_fp, sample_fn)
            eval_row['参数名-精确率'], eval_row['参数名-召回率'], eval_row['参数名-F1'] = precision, recall, f1

            # --- 评估逻辑 1: 宽松标准 (忽略值为"无"的可选参数) ---
            gt_optional_keys = {key for key, value in gt_params.items() if value == "无"}
            gt_required_keys = gt_keys - gt_optional_keys
            is_struct_correct_loose = (gt_required_keys.issubset(pred_keys)) and (pred_keys.issubset(gt_keys))

            if is_struct_correct_loose:
                eval_row['参数结构准确率(宽松)'] = 1
                all_closed_values_match_loose = True
                for key in pred_keys: 
                    if not is_open_domain(key, gt_tool, tools_def) and gt_params.get(key) != pred_params.get(key):
                        all_closed_values_match_loose = False
                        break
                if all_closed_values_match_loose:
                    eval_row['宽松综合准确率(必需参数)'] = 1
            
            # --- 评估逻辑 2: 严格标准 (参数名集合需完全匹配) ---
            if gt_keys == pred_keys:
                all_closed_values_match_strict = True
                for key in gt_keys:
                    if not is_open_domain(key, gt_tool, tools_def):
                        if gt_params.get(key) != pred_params.get(key):
                            all_closed_values_match_strict = False
                            break
                if all_closed_values_match_strict:
                    eval_row['严格综合准确率(全参数)'] = 1

            # --- 参数值评估（用于计算Fuzz, BLEU等） ---
            closed_correct, closed_total, open_fuzz, open_bleu, open_rouge = 0, 0, [], [], []
            for key in (gt_keys & pred_keys):
                if is_open_domain(key, gt_tool, tools_def):
                    pred_val, gt_val = pred_params.get(key, ""), gt_params.get(key, "")
                    open_fuzz.append(fuzz.ratio(pred_val, gt_val))
                    if pred_val and gt_val:
                        bleu_scores = calculate_bleu(pred_val, gt_val)
                        rouge_scores = calculate_rouge(pred_val, gt_val)
                        open_bleu.append(bleu_scores)
                        open_rouge.append(rouge_scores)
                else:
                    closed_total += 1
                    if gt_params.get(key) == pred_params.get(key):
                        closed_correct += 1
            
            if closed_total > 0: eval_row['闭域参数值-准确率'] = safe_div(closed_correct, closed_total)
            if open_fuzz: eval_row['开域参数值-Fuzz相似度'] = sum(open_fuzz) / len(open_fuzz)
            if open_bleu:
                eval_row['开域参数值-BLEU1'] = safe_div(sum(b['bleu_1'] for b in open_bleu), len(open_bleu), to_percent=False)
                eval_row['开域参数值-BLEU2'] = safe_div(sum(b['bleu_2'] for b in open_bleu), len(open_bleu), to_percent=False)
                eval_row['开域参数值-BLEU3'] = safe_div(sum(b['bleu_3'] for b in open_bleu), len(open_bleu), to_percent=False)
                eval_row['开域参数值-BLEU4'] = safe_div(sum(b['bleu_4'] for b in open_bleu), len(open_bleu), to_percent=False)
            if open_rouge:
                eval_row['开域参数值-ROUGE1'] = safe_div(sum(r['rouge_1'] for r in open_rouge), len(open_rouge), to_percent=False)
                eval_row['开域参数值-ROUGE2'] = safe_div(sum(r['rouge_2'] for r in open_rouge), len(open_rouge), to_percent=False)
                eval_row['开域参数值-ROUGEL'] = safe_div(sum(r['rouge_l'] for r in open_rouge), len(open_rouge), to_percent=False)
        
        detailed_results.append(eval_row)
    return detailed_results

def save_details_to_csv(detailed_results, filepath):
    """保存详细结果到CSV"""
    if not detailed_results: return
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    headers = list(detailed_results[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(detailed_results)
    print(f"\n详细评估结果已保存到: {filepath}")

def print_summary_report(detailed_results, report_title="模型综合"):
    """打印聚合评估报告"""
    total_samples = len(detailed_results)
    if total_samples == 0: 
        print(f"\n" + "="*22 + f" {report_title} 评估报告 " + "="*22)
        print("该类别下无样本。")
        print("="*62)
        return

    accuracy_format = safe_div(sum(r['格式正确率'] for r in detailed_results), total_samples)
    accuracy_combined_loose = safe_div(sum(r['宽松综合准确率(必需参数)'] for r in detailed_results), total_samples)
    accuracy_combined_strict = safe_div(sum(r['严格综合准确率(全参数)'] for r in detailed_results), total_samples)
    accuracy_dsl = safe_div(sum(r['完整匹配准确率'] for r in detailed_results), total_samples)
    
    func_correct_count = sum(r['函数名准确率'] for r in detailed_results)
    accuracy_func = safe_div(func_correct_count, total_samples)
    accuracy_pname_struct_loose = safe_div(sum(r['参数结构准确率(宽松)'] for r in detailed_results), func_correct_count) if func_correct_count > 0 else 0

    valid_prf_samples = [r for r in detailed_results if r['函数名准确率']]
    avg_precision_pname = safe_div(sum(r['参数名-精确率'] for r in valid_prf_samples), len(valid_prf_samples), to_percent=False) if valid_prf_samples else 0
    avg_recall_pname = safe_div(sum(r['参数名-召回率'] for r in valid_prf_samples), len(valid_prf_samples), to_percent=False) if valid_prf_samples else 0
    avg_f1_pname = safe_div(sum(r['参数名-F1'] for r in valid_prf_samples), len(valid_prf_samples), to_percent=False) if valid_prf_samples else 0

    closed_val_samples = [r for r in detailed_results if r['闭域参数值-准确率'] is not None]
    avg_cval_accuracy = safe_div(sum(r['闭域参数值-准确率'] for r in closed_val_samples), len(closed_val_samples), to_percent=False) if closed_val_samples else 0
    
    open_domain_samples = [r for r in detailed_results if r['开域参数值-Fuzz相似度'] is not None]
    avg_fuzz = safe_div(sum(r['开域参数值-Fuzz相似度'] for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    avg_bleu1 = safe_div(sum(r.get('开域参数值-BLEU1', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    avg_bleu2 = safe_div(sum(r.get('开域参数值-BLEU2', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    avg_bleu3 = safe_div(sum(r.get('开域参数值-BLEU3', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    avg_bleu4 = safe_div(sum(r.get('开域参数值-BLEU4', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    avg_rouge1 = safe_div(sum(r.get('开域参数值-ROUGE1', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    avg_rouge2 = safe_div(sum(r.get('开域参数值-ROUGE2', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    avg_rougeL = safe_div(sum(r.get('开域参数值-ROUGEL', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0
    
    print("\n" + "="*22 + f" {report_title} 评估报告 " + "="*22)
    print(f"总样本数: {total_samples}")
    print("\n--- [ 基础评估: 格式与匹配 ] ---")
    print(f"格式正确率:                   {accuracy_format:.2f}%")
    print(f"宽松综合准确率 (必需参数):     {accuracy_combined_loose:.2f}%")
    print(f"严格综合准确率 (全参数):       {accuracy_combined_strict:.2f}%")
    print(f"完整匹配准确率 (Full Match):    {accuracy_dsl:.2f}%")
    print("\n--- [ 级别 1: 函数名评估 ] ---")
    print(f"准确率: {accuracy_func:.2f}%  (正确: {func_correct_count})")
    print("\n--- [ 级别 2: 参数名评估 (当函数名正确时) ] ---")
    print(f"参数结构准确率 (宽松): {accuracy_pname_struct_loose:.2f}%")
    print(f"宏观平均-精确率:       {avg_precision_pname:.2f}%")
    print(f"宏观平均-召回率:       {avg_recall_pname:.2f}%")
    print(f"宏观平均-F1:           {avg_f1_pname:.2f}%")
    print("\n--- [ 级别 3: 参数值评估 (当参数名正确时) ] ---")
    print("\n  -- 闭域参数值 --")
    print(f"准确率: {avg_cval_accuracy:.2f}%")
    print("\n  -- 开域参数值 --")
    print(f"Avg. Fuzz相似度: {avg_fuzz:.2f}")
    print(f"Avg. BLEU-1/2/3/4: {avg_bleu1:.4f} / {avg_bleu2:.4f} / {avg_bleu3:.4f} / {avg_bleu4:.4f}")
    print(f"Avg. ROUGE-1/2/L:  {avg_rouge1:.4f} / {avg_rouge2:.4f} / {avg_rougeL:.4f}")
    print(f"(开域评估样本数: {len(open_domain_samples)})")
    print("\n" + "="*62)

def main():
    """主函数"""
    try:
        with open(INPUT_INFERENCE_PATH, 'r', encoding='utf-8') as f:
            inference_results = json.load(f)
        print(f"成功加载 {len(inference_results)} 条推理结果用于评估。")
    except Exception as e:
        print(f"错误: 加载推理结果文件 '{INPUT_INFERENCE_PATH}' 失败 - {e}")
        return
    
    detailed_results = run_evaluation_and_get_details(inference_results)
    save_details_to_csv(detailed_results, OUTPUT_EVALUATION_DETAILS_PATH)
    
    print_summary_report(detailed_results, "总体 (Overall)")
    
    all_tags = sorted(list(set(r.get('标签(tag)') for r in detailed_results if r.get('标签(tag)'))))
    for tag in all_tags:
        results_for_tag = [r for r in detailed_results if r.get('标签(tag)') == tag]
        print_summary_report(results_for_tag, f"标签: {tag}")

if __name__ == "__main__":
    main()