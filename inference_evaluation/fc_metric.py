import os
import re
import ast
import csv
import json
import jieba
import logging
from thefuzz import fuzz
from collections import Counter
from datetime import datetime

# --- 日志配置 ---
LOG_DIR = '../logs/metric'
MODEL_NAME = "fc0801_dapo"
date = "20250805"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_file = os.path.join(LOG_DIR, f'{date}_fc_{MODEL_NAME}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 文件路径配置 ---
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
INPUT_INFERENCE_PATH = f"/home/workspace/nl/FC/data/output/toolcall/lgq_4b/20250805/fc_evaluation_with_response_newform_fc0801_dapo_2025-08-05_10-01.json"
OUTPUT_EVALUATION_DETAILS_PATH = f"/home/workspace/nl/FC/data/output/toolcall/lgq_4b/{date}/fc_evaluation_results_{MODEL_NAME}_{timestamp}.csv"

# --- 1. 核心辅助函数 (部分已修改) ---
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
        param_matches = re.findall(r'(\w+)\s*=\s*(?:"(.*?)"|([^,)]+))', params_str)
        params = {key: val_quoted or val_unquoted for key, val_quoted, val_unquoted in param_matches}
        return tool_name, params
    except (ValueError, SyntaxError, IndexError): return None, {}

def parse_tools_from_prompt(user_prompt: str):
    tools_def = {}
    try:
        tools_match = re.search(r'<tools>(.*?)</tools>', user_prompt, re.DOTALL)
        if not tools_match: return {}
        tools_str = tools_match.group(1).strip()
        tool_list = ast.literal_eval(tools_str)
        for tool in tool_list:
            tool_name = tool.get('name')
            properties = tool.get('parameters', {}).get('properties', {})
            if tool_name:
                tools_def[tool_name] = properties
    except (Exception):
        return {}
    return tools_def

def is_open_domain(param_name: str, tool_name: str, tools_def: dict):
    if param_name == 'app': return False
    if tool_name in tools_def and param_name in tools_def[tool_name]:
        return 'enum' not in tools_def[tool_name].get(param_name, {})
    return True

def safe_div(num, den, to_percent=True):
    if den == 0: return 0.0
    result = num / den
    return result * 100 if to_percent else result

def calculate_prf1(tp, fp, fn):
    precision = safe_div(tp, tp + fp, to_percent=False)
    recall = safe_div(tp, tp + fn, to_percent=False)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision * 100, recall * 100, f1 * 100

def _get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def calculate_bleu(prediction_str, reference_str):
    if not prediction_str or not reference_str: return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
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
    if not prediction_str or not reference_str: return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
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
            '标签(tag)': result.get('tag', 'N/A'), '用户输入': result.get('user_prompt'),
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
        # if "在抖音中的商城店铺查找‘复古风家居装饰’按销量优先"in result.get("user_prompt", ""):
        #     print(f"{gt_params=}")
        #     print(f"{pred_params=}")
        #     print(result.get('model_output'))
        #     print(result.get("ground_truth", ""))

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

            pred_optional_keys = {key for key, value in pred_params.items() if value == "无" or value is None}
            pred_required_keys = pred_keys - pred_optional_keys

            # if gt_required_keys == pred_required_keys:
            if gt_keys == pred_keys:
                all_closed_values_match_strict = True
                for key in gt_keys:
                    if not is_open_domain(key, gt_tool, tools_def):
                        if gt_params.get(key) != pred_params.get(key):
                            all_closed_values_match_strict = False
                            break
                if all_closed_values_match_strict:
                    eval_row['严格综合准确率(全参数)'] = 1

            closed_correct, closed_total, open_fuzz, open_bleu, open_rouge = 0, 0, [], [], []
            for key in (gt_keys & pred_keys):
                is_open = is_open_domain(key, gt_tool, tools_def)
                pred_val, gt_val = pred_params.get(key, ""), gt_params.get(key, "")

                if is_open:
                    open_fuzz.append(fuzz.ratio(pred_val, gt_val))
                    if pred_val and gt_val:
                        bleu_scores = calculate_bleu(pred_val, gt_val)
                        rouge_scores = calculate_rouge(pred_val, gt_val)
                        open_bleu.append(bleu_scores)
                        open_rouge.append(rouge_scores)
                else:
                    closed_total += 1
                    if gt_val == pred_val:
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

# --- 3. 报告与保存函数 ---
def save_details_to_csv(detailed_results, filepath):
    if not detailed_results: return
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    headers = list(detailed_results[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(detailed_results)
    # **日志记录保存结果**
    logger.info(f"\n详细评估结果已保存到: {filepath}")

def print_summary_report(detailed_results, report_title="模型综合"):
    total_samples = len(detailed_results)
    if total_samples == 0:
        logger.info("\n" + "="*22 + f" {report_title} 评估报告 " + "="*22)
        logger.info("该类别下无样本。")
        logger.info("="*62)
        return

    accuracy_format = safe_div(sum(r['格式正确率'] for r in detailed_results), total_samples)
    accuracy_combined_loose = safe_div(sum(r['宽松综合准确率(必需参数)'] for r in detailed_results), total_samples)
    accuracy_combined_strict = safe_div(sum(r['严格综合准确率(全参数)'] for r in detailed_results), total_samples)
    accuracy_dsl = safe_div(sum(r['完整匹配准确率'] for r in detailed_results), total_samples)

    func_correct_count = sum(r['函数名准确率'] for r in detailed_results)
    accuracy_func = safe_div(func_correct_count, total_samples)
    accuracy_pname_struct_loose = safe_div(sum(r['参数结构准确率(宽松)'] for r in detailed_results), func_correct_count) if func_correct_count > 0 else 0.0

    valid_prf_samples = [r for r in detailed_results if r['函数名准确率']]
    avg_precision_pname = safe_div(sum(r['参数名-精确率'] for r in valid_prf_samples), len(valid_prf_samples), to_percent=False) if valid_prf_samples else 0.0
    avg_recall_pname = safe_div(sum(r['参数名-召回率'] for r in valid_prf_samples), len(valid_prf_samples), to_percent=False) if valid_prf_samples else 0.0
    avg_f1_pname = safe_div(sum(r['参数名-F1'] for r in valid_prf_samples), len(valid_prf_samples), to_percent=False) if valid_prf_samples else 0.0

    closed_val_samples = [r for r in detailed_results if r['闭域参数值-准确率'] is not None]
    avg_cval_accuracy = safe_div(sum(r['闭域参数值-准确率'] for r in closed_val_samples), len(closed_val_samples), to_percent=False) if closed_val_samples else 0.0

    open_domain_samples = [r for r in detailed_results if r['开域参数值-Fuzz相似度'] is not None]
    avg_fuzz = safe_div(sum(r['开域参数值-Fuzz相似度'] for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0

    avg_bleu1 = safe_div(sum(r.get('开域参数值-BLEU1', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0
    avg_bleu2 = safe_div(sum(r.get('开域参数值-BLEU2', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0
    avg_bleu3 = safe_div(sum(r.get('开域参数值-BLEU3', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0
    avg_bleu4 = safe_div(sum(r.get('开域参数值-BLEU4', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0
    avg_rouge1 = safe_div(sum(r.get('开域参数值-ROUGE1', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0
    avg_rouge2 = safe_div(sum(r.get('开域参数值-ROUGE2', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0
    avg_rougeL = safe_div(sum(r.get('开域参数值-ROUGEL', 0) for r in open_domain_samples), len(open_domain_samples), to_percent=False) if open_domain_samples else 0.0

    logger.info("\n" + "="*22 + f" {report_title} 评估报告 " + "="*22)
    logger.info(f"总样本数: {total_samples}")
    logger.info("\n--- [ 基础评估: 格式与匹配 ] ---")
    logger.info(f"格式正确率:                   {accuracy_format:.2f}%")
    logger.info(f"宽松综合准确率 (必需参数):     {accuracy_combined_loose:.2f}%")
    logger.info(f"严格综合准确率 (全参数):       {accuracy_combined_strict:.2f}%")
    logger.info(f"完整匹配准确率 (Full Match):   {accuracy_dsl:.2f}%")
    logger.info("\n--- [ 级别 1: 函数名评估 ] ---")
    logger.info(f"准确率: {accuracy_func:.2f}%  (正确: {func_correct_count})")
    logger.info("\n--- [ 级别 2: 参数名评估 (当函数名正确时) ] ---")
    logger.info(f"参数结构准确率 (宽松): {accuracy_pname_struct_loose:.2f}%")
    logger.info(f"宏观平均-精确率:       {avg_precision_pname:.2f}%")
    logger.info(f"宏观平均-召回率:       {avg_recall_pname:.2f}%")
    logger.info(f"宏观平均-F1:           {avg_f1_pname:.2f}%")
    logger.info("\n--- [ 级别 3: 参数值评估 (当参数名正确时) ] ---")
    logger.info("\n  -- 闭域参数值 --")
    logger.info(f"准确率: {avg_cval_accuracy:.2f}%")
    logger.info("\n  -- 开域参数值 --")
    logger.info(f"Avg. Fuzz相似度: {avg_fuzz:.2f}")
    logger.info(f"Avg. BLEU-1/2/3/4: {avg_bleu1:.4f} / {avg_bleu2:.4f} / {avg_bleu3:.4f} / {avg_bleu4:.4f}")
    logger.info(f"Avg. ROUGE-1/2/L:  {avg_rouge1:.4f} / {avg_rouge2:.4f} / {avg_rougeL:.4f}")
    logger.info(f"(开域评估样本数: {len(open_domain_samples)})")
    logger.info("\n" + "="*62)

def main():
    try:
        with open(INPUT_INFERENCE_PATH, 'r', encoding='utf-8') as f:
            new_format_data = json.load(f)
        logger.info(f"成功加载 {len(new_format_data)} 条新格式推理结果用于评估。")
    except Exception as e:
        logger.error(f"错误: 加载推理结果文件 '{INPUT_INFERENCE_PATH}' 失败 - {e}")
        return

    inference_results_adapted = []
    for item in new_format_data:
        system_prompt = ""
        user_content = ""
        for message in item.get("messages", []):
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
            elif message.get("role") == "user":
                user_content = message.get("content", "")
        full_prompt = system_prompt + user_content if system_prompt.endswith("任务是:\"") else system_prompt + "\n用户的问题或任务是:\"" + user_content + "\""
        adapted_item = {
            "user_prompt": full_prompt,
            "ground_truth": item.get("label"),
            "model_output": item.get("predict").replace('"', ""),
            "delay_seconds": -1.0,
            "tag": item.get("tag")
        }
        inference_results_adapted.append(adapted_item)

    logger.info(f"已将 {len(inference_results_adapted)} 条数据成功适配，开始评估...")

    detailed_results = run_evaluation_and_get_details(inference_results_adapted)
    save_details_to_csv(detailed_results, OUTPUT_EVALUATION_DETAILS_PATH)

    print_summary_report(detailed_results, "总体 (Overall)")

    all_tags = sorted(list(set(r.get('标签(tag)') for r in detailed_results if r.get('标签(tag)'))))
    if len(all_tags) > 1:
        for tag in all_tags:
            results_for_tag = [r for r in detailed_results if r.get('标签(tag)') == tag]
            print_summary_report(results_for_tag, f"标签: {tag}")

if __name__ == "__main__":
    main()