import pandas as pd
import os
import json
import ast
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
import time
import itertools  
from sklearn.model_selection import train_test_split 

# --- 召回器和评测模块 ---
from rank_bm25 import BM25Okapi
import jieba
from sklearn.metrics import roc_auc_score


class ToolRetriever:
    def __init__(self, all_tools_corpus: list, all_tools_definitions: list, k1=1.5, b=0.75):
        self.tool_corpus = all_tools_corpus
        self.tool_definitions = all_tools_definitions
        self.k1 = k1
        self.b = b
        
        # 增加jieba自定义词典，可以提高对工具名称和核心业务词的分词准确性
        self._add_jieba_words()
        
        tokenized_corpus = [self._tokenize(doc) for doc in self.tool_corpus]
        # 使用传入的参数初始化 BM25
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def _tokenize(self, text: str) -> list[str]:
        return jieba.lcut(text, cut_all=False)

    def _add_jieba_words(self):
        # 添加工具名
        for tool in self.tool_definitions:
            jieba.add_word(tool.get('name', ''), freq=100)
        # 添加核心业务词
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "降价", "签到", "积分", "发票", "开票", "报销凭证"]
        for word in core_words:
            jieba.add_word(word, freq=100)

    def retrieve_with_scores(self, query: str, top_k: int):
        tokenized_query = self._tokenize(query)
        all_scores = self.bm25.get_scores(tokenized_query)
        
        top_k_indices = all_scores.argsort()[-top_k:][::-1]
        retrieved = [self.tool_definitions[i] for i in top_k_indices if all_scores[i] > 0]
        
        return retrieved, all_scores

def _get_tool_names(tools: list) -> set:
    if not isinstance(tools, list): return set()
    return {tool.get('name') for tool in tools}

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    return len(retrieved_names_at_k.intersection(ground_truth_names)) / len(ground_truth_names)

def calculate_completeness_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    return 1.0 if ground_truth_names.issubset(retrieved_names_at_k) else 0.0

def calculate_ndcg_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, tool in enumerate(retrieved[:k]) if tool.get('name') in ground_truth_names)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth_names), k)))
    return dcg / idcg if idcg > 0 else 0.0

def calculate_hit_ratio_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    retrieved_names = _get_tool_names(retrieved[:k])
    return 1.0 if retrieved_names & _get_tool_names(ground_truth) else 0.0

def calculate_average_precision_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    gt_names = _get_tool_names(ground_truth)
    if not gt_names: return 1.0
    hit_count = 0
    sum_prec = 0.0
    for i, tool in enumerate(retrieved[:k]):
        if tool.get('name') in gt_names:
            hit_count += 1
            sum_prec += hit_count / (i + 1)
    return sum_prec / len(gt_names)

def calculate_mrr_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    gt_names = _get_tool_names(ground_truth)
    for i, tool in enumerate(retrieved[:k]):
        if tool.get('name') in gt_names:
            return 1.0 / (i + 1)
    return 0.0

def calculate_auc_for_query(all_scores: np.ndarray, tool_defs: list, ground_truth: list) -> float:
    gt_names = _get_tool_names(ground_truth)
    labels = [1 if t['name'] in gt_names else 0 for t in tool_defs]
    try:
        if len(set(labels)) < 2: return 0.5
        return roc_auc_score(labels, all_scores)
    except ValueError: return 0.5


# --- 工具定义模块 ---
def get_exact_tool_definitions():
    tools = [
        {"name": "open_orders_bought(app, order_status)", "description": "在app应用程序中查看买入的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_sold(app, order_status)", "description": "在app应用程序中查看自己售卖的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_all_review(app)", "description": "在app应用程序中查看待评价状态的订单列表，在不指定购买还是售卖的订单时，及全都要看时使用。"},
        {"name": "search_order(app, search_info, order_status)", "description": "在app应用程序中搜索订单"},
        {"name": "open_invoice_page(app, page_type)", "description": "在app应用程序中打开与发票相关的页面"},
        {"name": "open_cart_content(app, filter_type)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）指定类型的商品"},
        {"name": "search_cart_content(app, search_info)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）查找商品"},
        {"name": "open_customer_service(app)", "description": "在app应用程序中联系客服"},
        {"name": "sign_in(app, page_type)", "description": "在app程序中完成每日签到，领取积分、金币等奖励的操作"},
        {"name": "open_favorite_goods(app, filter_type, order_type)", "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面，并按照条件进行筛选"},
        {"name": "open_favorite_stores(app, filter_type)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并按照条件进行筛选"},
        {"name": "search_in_favorite_goods(app, search_info)", "description": "在app程序中打开收藏的、喜爱、想要或关注商品的页面，并在其中的搜索栏中进行搜索"},
        {"name": "search_in_favorite_stores(app, search_info)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并在其中的搜索栏搜索商品"},
        {"name": "search_goods(app, search_info, order_type)", "description": "在app程序中依据名称搜索商品，可以指定搜索结果的排序方式"},
        {"name": "search_stores(app, search_info, filter_type, order_type)", "description": "在app程序中依据名称搜索店铺，可以使用筛选器限制搜索结果，也可以指定搜索结果的排序方式"},
        {"name": "open_search_history(app)", "description": "打开app程序的搜索历史界面"},
        {"name": "delete_search_history(app)", "description": "清除app中的搜索历史"},
        {"name": "open_camera_search(app)", "description": "打开app程序的图片搜索功能"},
        {"name": "open_logistics_receive(app, filter_type)", "description": "打开显示已购商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_logistics_send(app, filter_type)", "description": "打开显示已售商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_express_delivery(app)", "description": "打开app寄送快递的界面"},
        {"name": "open_app(app)", "description": "打开指定的应用程序"},
    ]
    return tools


# 定义同义词扩展函数和语料库构建函数
def get_synonyms():
    return {
        "购物车": ["采购车"],
        "收藏": ["喜欢", "想要", "关注", "收藏夹"],
        "查": ["找", "搜", "搜索", "查询"], 
        "找订单": ["查订单", "搜订单"],
        "看物流": ["查快递", "包裹进度"],
        "买的东西": ["我买的", "收到的", "购买记录"],
        "卖的东西": ["我卖的", "发出的", "售出记录"],
        "发票": ["开票", "报销凭证"],
    }

def build_corpus(data_df: pd.DataFrame, tool_definitions: list) -> list:
    print("--- 正在构建增强语料库... ---")
    synonyms = get_synonyms()
    
    tool_text_aggregator = defaultdict(list)
    for _, row in data_df.iterrows():
        if row['available_tools']:
            tool_name = row['available_tools'][0]['name']
            tool_text_aggregator[tool_name].append(row['instruction_template'])
            tool_text_aggregator[tool_name].append(row['final_query'])
            
    all_tools_corpus = []
    for tool_def in tool_definitions:
        tool_name = tool_def['name']
        
        # 基础文本：描述 + 聚合的用户查询
        description = tool_def.get('description', '')
        aggregated_queries = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
        
        # 同义词扩展
        synonym_expansion = []
        for word, syns in synonyms.items():
            if word in description:
                synonym_expansion.extend(syns)
        
        synonym_text = ' '.join(set(synonym_expansion))
        
        # 构建最终文档
        document = f"这是一个工具，它的功能是 {description}。同义词参考: {synonym_text}。用户可能会这样说：{aggregated_queries}"

        # document += f" 同义词参考: {synonym_text}。用户可能会这样说：{aggregated_queries}"
        all_tools_corpus.append(document)
        print(all_tools_corpus)
        
    print(f"语料库构建完成，共 {len(all_tools_corpus)} 个工具。\n")
    return all_tools_corpus

# --- 主程序 ---
def main():
    # --- 0. 配置区域 ---
    # annotated_data_file_path = r'D:\Agent\data\corrected_data.csv' 
    annotated_data_file_path = r'D:\Agent\data\单gt.csv' 
    K_VALUES = [1, 2, 3, 4, 5]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10 # 打印多少个错误案例

    # --- 1. 数据加载与划分 ---
    print("--- 步骤 1: 加载已标注数据并划分 ---")
    try:
        required_columns = ['instruction_template', 'final_query', 'is_train', 'available_tools']
        data_df = pd.read_csv(annotated_data_file_path, usecols=required_columns)
    except Exception as e:
        print(f"错误: 读取文件 '{annotated_data_file_path}' 失败. {e}")
        return
    
    def parse_tools(tool_string):
        try: return ast.literal_eval(tool_string)
        except (ValueError, SyntaxError): return []
    data_df['available_tools'] = data_df['available_tools'].apply(parse_tools)
    
    full_train_df = data_df[data_df['is_train'] == 0].copy()
    test_df = data_df[data_df['is_train'] == 1].copy()
    
    # 从 full_train_df 中划分出验证集，用于参数搜索 
    if len(full_train_df) > 10: # 确保有足够数据划分
        train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)
    else:
        train_df, val_df = full_train_df, full_train_df # 数据太少，不划分

    print(f"数据划分完成：训练集 {len(train_df)} 条，验证集 {len(val_df)} 条，测试集 {len(test_df)} 条。\n")

    # --- 2. 参数搜索 (Grid Search) ---
    print("--- 步骤 2: 在验证集上搜索最佳 BM25 参数 ---")
    all_tools_definitions = get_exact_tool_definitions()
    
    # 使用训练集构建语料库
    corpus_for_tuning = build_corpus(train_df, all_tools_definitions)
    
    best_score = -1
    best_params = {'k1': 1.5, 'b': 0.75} # 默认值
    k1_range = [1.2, 1.5, 1.8, 2.0]
    b_range = [0.6, 0.75, 0.9]

    for k1, b in tqdm(list(itertools.product(k1_range, b_range)), desc="参数搜索中"):
        temp_retriever = ToolRetriever(corpus_for_tuning, all_tools_definitions, k1=k1, b=b)
        recalls_at_1 = []
        for _, row in val_df.iterrows():
            query = row['final_query']
            gt = row['available_tools']
            retrieved, _ = temp_retriever.retrieve_with_scores(query, top_k=1)
            recalls_at_1.append(calculate_recall_at_k(retrieved, gt, k=1))
        
        current_score = np.mean(recalls_at_1)
        if current_score > best_score:
            best_score = current_score
            best_params = {'k1': k1, 'b': b}

    print(f"参数搜索完成！最佳参数: {best_params} (在验证集上的 Recall@1: {best_score:.4f})\n")

    # --- 3. 构建最终召回器 ---
    print("--- 步骤 3: 使用最佳参数和全部训练数据构建最终召回器 ---")
    # 使用全部训练数据 (train + val) 构建最终语料库
    final_corpus = build_corpus(full_train_df, all_tools_definitions)
    # 使用最佳参数构建最终召回器
    retriever = ToolRetriever(final_corpus, all_tools_definitions, k1=best_params['k1'], b=best_params['b'])
    print("最终召回器构建完成。\n")

    # --- 4. 在测试集上评测 ---
    print(f"--- 步骤 4: 开始在 {len(test_df)} 个测试集样本上进行评测 ---")
    results = {
        'Recall@K': {k: [] for k in K_VALUES},
        'HR@K': {k: [] for k in K_VALUES},
        'MAP@K': {k: [] for k in K_VALUES},
        'MRR@K': {k: [] for k in K_VALUES},
        'NDCG@K': {k: [] for k in K_VALUES},
        'COMP@K': {k: [] for k in K_VALUES},
        'AUC': [],
        'processing_time': []
    }
    error_cases = []

    for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="测试集评测中")):
        query = row['final_query']
        query = row["instruction_template"]
        ground_truth = row['available_tools']
        
        start_time = time.perf_counter()
        retrieved, all_scores = retriever.retrieve_with_scores(query, top_k=max(K_VALUES))
        duration = time.perf_counter() - start_time
        
        results['processing_time'].append(duration)
        results['AUC'].append(calculate_auc_for_query(all_scores, all_tools_definitions, ground_truth))

        for k in K_VALUES:
            results['Recall@K'][k].append(calculate_recall_at_k(retrieved, ground_truth, k))
            results['HR@K'][k].append(calculate_hit_ratio_at_k(retrieved, ground_truth, k))
            results['MAP@K'][k].append(calculate_average_precision_at_k(retrieved, ground_truth, k))
            results['MRR@K'][k].append(calculate_mrr_at_k(retrieved, ground_truth, k))
            results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved, ground_truth, k))
            results['COMP@K'][k].append(calculate_completeness_at_k(retrieved, ground_truth, k))
        
        # 错误分析：记录 Top-1 预测失败的样本 
        is_top1_correct = calculate_recall_at_k(retrieved, ground_truth, k=1) >= 1.0
        if not is_top1_correct:
            gt_name = _get_tool_names(ground_truth).pop() if ground_truth else "N/A"
            pred_name_top1 = retrieved[0].get('name') if retrieved else "N/A"
            error_cases.append({
                "Query": query,
                "Ground Truth": [gt_name],
                "Prediction@1": [pred_name_top1],
                "Prediction@5": [r.get('name') for r in retrieved]
            })

    # --- 5. 汇总并报告结果 ---
    print("\n\n--- 步骤 5: 评测结果报告 ---")
    final_scores = {}
    for metric, vals in results.items():
        if metric == 'AUC': final_scores['AUC'] = np.mean(vals)
        elif metric == 'processing_time': continue
        else: final_scores[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    report_df = pd.DataFrame({ m: final_scores[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    print("BM25 召回模型在测试集上的评测结果:")
    print("-" * 50)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores['AUC']:.4f}")
    print("-" * 50)
    
    total_time, avg_time_ms = np.sum(results['processing_time']), np.mean(results['processing_time']) * 1000
    qps = len(test_df) / total_time if total_time > 0 else 0
    print("\n性能评测:")
    print("-" * 50)
    print(f"测试样本总数: {len(test_df)} 条")
    print(f"总耗时: {total_time:.4f} 秒, 平均每条耗时: {avg_time_ms:.4f} 毫秒, QPS: {qps:.2f}")
    print("-" * 50)
    
    # 5. 打印错误分析报告 
    print(f"\n\n--- 步骤 6: Top-1 错误案例分析 (共 {len(error_cases)} 个错误) ---")
    if not error_cases:
        print("🎉 恭喜！在测试集上没有发现 Top-1 错误案例！")
    else:
        for i, case in enumerate(error_cases[:NUM_ERROR_EXAMPLES_TO_PRINT]):
            print(f"\n--- 错误案例 {i+1}/{len(error_cases)} ---")
            print(f"  [查询 Query]: {case['Query']}")
            print(f"  [真实工具 Ground Truth]: {case['Ground Truth']}")
            print(f"  [预测工具 Prediction@1]: {case['Prediction@1']}")
            print(f"  [预测工具 Prediction@5]: {case['Prediction@5']}")
        if len(error_cases) > NUM_ERROR_EXAMPLES_TO_PRINT:
            print(f"\n... (仅显示前 {NUM_ERROR_EXAMPLES_TO_PRINT} 个错误案例) ...")
    print("-" * 50)


if __name__ == "__main__":
    main()