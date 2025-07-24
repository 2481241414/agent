# -*- coding: utf-8 -*-

# --- 核心依赖 ---
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

# ★★★ 新增：导入布隆过滤器 ★★★
# 请先安装: pip install pybloom-live
from pybloom_live import BloomFilter


# ★★★ 1. 将分词器和布隆过滤器整合到 ToolRetriever 中 ★★★
class ToolRetriever:
    def __init__(self, all_tools_corpus: list, all_tools_definitions: list, k1=1.5, b=0.75):
        self.tool_corpus = all_tools_corpus
        self.tool_definitions = all_tools_definitions
        self.k1 = k1
        self.b = b
        
        # 增加jieba自定义词典，可以提高对工具名称和核心业务词的分词准确性
        self._add_jieba_words()
        
        # 步骤 1: 对语料库进行分词
        tokenized_corpus = [self._tokenize(doc) for doc in self.tool_corpus]
        
        # 步骤 2: 使用分词后的语料库初始化 BM25
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        # ★★★ 新增：步骤 3: 构建布隆过滤器 ★★★
        # 将所有词元（token）添加到布隆过滤器中，用于快速预筛选
        print("    -> 正在构建布隆过滤器...")
        # 提取所有不重复的词元
        all_tokens = set(itertools.chain.from_iterable(tokenized_corpus))
        # 根据词元数量和期望的误报率初始化布隆过滤器
        # 误报率设置得较低(0.1%)，以确保不会错误地过滤掉太多相关查询
        self.bloom_filter = BloomFilter(capacity=len(all_tokens) + 100, error_rate=0.001)
        for token in all_tokens:
            self.bloom_filter.add(token)
        print(f"    -> 布隆过滤器构建完成，包含 {len(all_tokens)} 个独立词元。")


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

        # ★★★ 新增：布隆过滤器预检 ★★★
        # 如果查询中的所有词元都确定不在语料库的词汇表中，直接返回空结果，避免无效计算。
        # 这里使用 any()，如果查询中至少有一个词元可能存在于布隆过滤器中，则继续。
        # 如果所有词元都不存在（not any(...)），则提前返回。
        if tokenized_query and not any(token in self.bloom_filter for token in tokenized_query):
            # 构造一个与正常流程兼容的空结果
            empty_scores = np.zeros(len(self.tool_definitions))
            return [], empty_scores

        # 如果通过了布隆过滤器检查，则执行BM25计算
        all_scores = self.bm25.get_scores(tokenized_query)
        
        # 筛选出分数大于0的工具
        positive_score_indices = np.where(all_scores > 0)[0]
        
        # 如果没有正分数的工具，直接返回空
        if len(positive_score_indices) == 0:
            return [], all_scores

        # 在有正分数的工具中，找到top_k
        top_k_indices_in_positive = np.argsort(all_scores[positive_score_indices])[-top_k:][::-1]
        top_k_indices = positive_score_indices[top_k_indices_in_positive]
        
        retrieved = [self.tool_definitions[i] for i in top_k_indices]
        
        return retrieved, all_scores

# --- (所有评测函数保持不变) ---
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
    gt_names = _get_tool_names(ground_truth)
    if not gt_names: return 1.0
    return 1.0 if retrieved_names & gt_names else 0.0

def calculate_average_precision_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    gt_names = _get_tool_names(ground_truth)
    if not gt_names: return 1.0
    hit_count = 0
    sum_prec = 0.0
    for i, tool in enumerate(retrieved[:k]):
        if tool.get('name') in gt_names:
            hit_count += 1
            sum_prec += hit_count / (i + 1)
    return sum_prec / len(gt_names) if gt_names else 0.0

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


# --- 权威工具定义模块 (保持不变) ---
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


# --- 同义词扩展函数和语料库构建函数 (保持不变) ---
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
        # 确保 available_tools 是一个列表且不为空
        if row['available_tools'] and isinstance(row['available_tools'], list):
            # 单gt数据，直接取第一个工具名
            tool_name = row['available_tools'][0]['name']
            tool_text_aggregator[tool_name].append(row['instruction_template'])
            tool_text_aggregator[tool_name].append(row['final_query'])
            
    all_tools_corpus = []
    for tool_def in tool_definitions:
        tool_name = tool_def['name']
        
        description = tool_def.get('description', '')
        aggregated_queries = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
        
        synonym_expansion = []
        for word, syns in synonyms.items():
            if word in description:
                synonym_expansion.extend(syns)
        
        synonym_text = ' '.join(set(synonym_expansion))
        
        document = f"这是一个工具，它的功能是 {description}。同义词参考: {synonym_text}。用户可能会这样说：{aggregated_queries}"
        all_tools_corpus.append(document)
        
    print(f"语料库构建完成，共 {len(all_tools_corpus)} 个工具。\n")
    return all_tools_corpus

# --- 主程序 (保持不变) ---
def main():
    # --- 0. 配置区域 ---
    # annotated_data_file_path = r'D:\Agent\data\corrected_data.csv' # 请确保路径正确
    annotated_data_file_path = r'D:\Agent\data\单gt.csv' # 请确保路径正确
    K_VALUES = [1, 2, 3, 4, 5]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10

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
    
    if len(full_train_df) > 10:
        train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)
    else:
        train_df, val_df = full_train_df, full_train_df

    print(f"数据划分完成：训练集 {len(train_df)} 条，验证集 {len(val_df)} 条，测试集 {len(test_df)} 条。\n")

    # --- 2. 参数搜索 (Grid Search) ---
    print("--- 步骤 2: 在验证集上搜索最佳 BM25 参数 ---")
    all_tools_definitions = get_exact_tool_definitions()
    
    corpus_for_tuning = build_corpus(train_df, all_tools_definitions)
    
    best_score = -1
    best_params = {'k1': 1.5, 'b': 0.75}
    k1_range = [1.2, 1.5, 1.8, 2.0]
    b_range = [0.6, 0.75, 0.9]

    for k1, b in tqdm(list(itertools.product(k1_range, b_range)), desc="参数搜索中"):
        # 每次循环都创建一个新的、临时的ToolRetriever实例
        temp_retriever = ToolRetriever(corpus_for_tuning, all_tools_definitions, k1=k1, b=b)
        recalls_at_1 = []
        for _, row in val_df.iterrows():
            # query = row['final_query']
            query = row['instruction_templete']
            gt = row['available_tools']
            retrieved, _ = temp_retriever.retrieve_with_scores(query, top_k=1)
            recalls_at_1.append(calculate_recall_at_k(retrieved, gt, k=1))
        
        current_score = np.mean(recalls_at_1)
        if current_score > best_score:
            best_score = current_score
            best_params = {'k1': k1, 'b': b}

    print(f"\n参数搜索完成！最佳参数: {best_params} (在验证集上的 Recall@1: {best_score:.4f})\n")

    # --- 3. 构建最终召回器 ---
    print("--- 步骤 3: 使用最佳参数和全部训练数据构建最终召回器 ---")
    final_corpus = build_corpus(full_train_df, all_tools_definitions)
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
        # query = row["instruction_template"] # 您可以切换使用哪个字段作为查询
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
    
    # --- 6. 错误分析报告 ---
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