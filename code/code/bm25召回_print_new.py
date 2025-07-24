import pandas as pd
import os
import json
import ast
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict

# --- 召回器和评测模块 (无需修改) ---
from rank_bm25 import BM25Okapi
import jieba

class ToolRetriever:
    # ... (这部分代码和之前一样，无需修改) ...
    def __init__(self, all_tools_corpus: list, all_tools_definitions: list):
        self.tool_corpus = all_tools_corpus
        self.tool_definitions = all_tools_definitions
        tokenized_corpus = [self._tokenize(doc) for doc in self.tool_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    def _tokenize(self, text: str) -> list[str]: return jieba.lcut(text)
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = doc_scores.argsort()[-top_k:][::-1]
        return [self.tool_definitions[i] for i in top_k_indices if doc_scores[i] > 0]

# ... (评测函数也和之前一样，无需修改) ...
def _get_tool_names(tools: list) -> set:
    if not isinstance(tools, list): return set()
    return {tool.get('name') for tool in tools}

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    return len(retrieved_names_at_k.intersection(ground_truth_names)) / len(ground_truth_names)

def calculate_completeness_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    return 1.0 if ground_truth_names.issubset(retrieved_names_at_k) else 0.0

def calculate_ndcg_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, tool in enumerate(retrieved[:k]) if tool.get('name') in ground_truth_names)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth_names), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★    新增模块：权威的工具定义知识库    ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def get_exact_tool_definitions():
    """
    提供一个权威的、唯一的工具定义来源。
    这里的 description 应该是工具本身固有的、不包含用户查询的描述。
    """
    tools = [
        # --- 订单(124) ---
        {"name": "open_orders_bought(app, order_status)", "description": "在app应用程序中查看买入的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_sold(app, order_status)", "description": "在app应用程序中查看自己售卖的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_all_review(app)", "description": "在app应用程序中查看待评价状态的订单列表，在不指定购买还是售卖的订单时，及全都要看时使用。"},
        {"name": "search_order(app, search_info, order_status)", "description": "在app应用程序中搜索订单"},
        # --- 发票(17) ---
        {"name": "open_invoice_page(app, page_type)", "description": "在app应用程序中打开与发票相关的页面"},
        # --- 购物车(37) ---
        {"name": "open_cart_content(app, filter_type)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）指定类型的商品"},
        {"name": "search_cart_content(app, search_info)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）查找商品"},
        # --- 客服(14) ---
        {"name": "open_customer_service(app)", "description": "在app应用程序中联系客服"},
        # --- 签到(16) ---
        {"name": "sign_in(app, page_type)", "description": "在app程序中完成每日签到，领取积分、金币等奖励的操作"},
        # --- 收藏(72) ---
        {"name": "open_favorite_goods(app, filter_type, order_type)", "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面，并按照条件进行筛选"},
        {"name": "open_favorite_stores(app, filter_type)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并按照条件进行筛选"},
        {"name": "search_in_favorite_goods(app, search_info)", "description": "在app程序中打开收藏的、喜爱、想要或关注商品的页面，并在其中的搜索栏中进行搜索"},
        {"name": "search_in_favorite_stores(app, search_info)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并在其中的搜索栏搜索商品"},
        # --- 搜索(146) ---
        {"name": "search_goods(app, search_info, order_type)", "description": "在app程序中依据名称搜索商品，可以指定搜索结果的排序方式"},
        {"name": "search_stores(app, search_info, filter_type, order_type)", "description": "在app程序中依据名称搜索店铺，可以使用筛选器限制搜索结果，也可以指定搜索结果的排序方式"},
        {"name": "open_search_history(app)", "description": "打开app程序的搜索历史界面"},
        {"name": "delete_search_history(app)", "description": "清除app中的搜索历史"},
        {"name": "open_camera_search(app)", "description": "打开app程序的图片搜索功能"},
        # --- 物流(68) ---
        {"name": "open_logistics_receive(app, filter_type)", "description": "打开显示已购商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_logistics_send(app, filter_type)", "description": "打开显示已售商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_express_delivery(app)", "description": "打开app寄送快递的界面"},
        # --- 启动 ---
        {"name": "open_app(app)", "description": "打开指定的应用程序"},
    ]
    return tools


# --- 主程序 ---
def main():
    # --- 0. 配置区域 ---
    data_file_path = r'D:\Agent\data\generated_queries - 0704手动筛选后.csv'
    mapping_file_path = r'D:\Agent\data\大类-工具映射关系表-0707-Cleaned.csv'
    K_VALUES = [1, 3, 5]
    NUM_EXAMPLES_TO_PRINT = 1

    # --- 1. 数据准备 ---
    print("--- 步骤 1: 准备评测数据 ---")
    try:
        required_columns = ['category', 'app_name', 'instruction_template', 'final_query', 'is_train']
        data_df = pd.read_csv(data_file_path, usecols=required_columns)
        mapping_df = pd.read_csv(mapping_file_path)
    except Exception as e:
        print(f"错误: 读取文件失败. {e}")
        return
    print("评测数据和映射表加载成功。\n")

    print("--- 步骤 2: 构建 BM25 语料库 (仅使用工具的权威描述) ---")

    # 1. 从 "工具说明书" 获取所有工具的精确定义
    all_tools_definitions = get_exact_tool_definitions()

    # 2. 构建用于BM25匹配的语料库 (document)
    #    文档内容 = 工具名 + 工具的描述
    all_tools_corpus = []
    for tool in all_tools_definitions:
        document = f"{tool.get('name', '')} {tool.get('description', '')}"
        all_tools_corpus.append(document)
        
    print(f"语料库构建完成，共 {len(all_tools_corpus)} 个工具。")
    # 打印第一个工具的文档，以供检查
    print("示例 - 第一个工具的文档内容:", all_tools_corpus[0])
    print("\n")


    # --- 3. 构建召回器 ---
    print("--- 步骤 3: 构建 BM25 召回器 ---")
    # 使用刚刚构建好的、干净的语料库来初始化召回器
    retriever = ToolRetriever(all_tools_corpus, all_tools_definitions)
    print("召回器构建完成。\n")

    # --- 4. 准备 Ground Truth 并评测 ---
    print(f"--- 步骤 4: 开始在 {len(data_df)} 个样本上进行评测 (K={K_VALUES}) ---")
    
    def get_ground_truth_tools(row, mapping_df):
        filtered_df = mapping_df[
            (mapping_df['app'] == row['app_name']) &
            (mapping_df['大类'] == row['category'])
        ]
        return [{'name': name} for name in filtered_df['function_name'].unique().tolist()]

    eval_df = data_df.copy()
    eval_df['available_tools'] = eval_df.apply(get_ground_truth_tools, axis=1, args=(mapping_df,))
    eval_df = eval_df[eval_df['available_tools'].apply(len) > 0]

    results = {metric: {k: [] for k in K_VALUES} for metric in ['Recall@K', 'NDCG@K', 'COMP@K']}
    
    for i, (_, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc="评测中")):
        query = row['final_query']
        ground_truth = row['available_tools']
        max_k = max(K_VALUES)
        retrieved_tools = retriever.retrieve(query, top_k=max_k)
        
        # 打印逻辑保持不变
        if i < NUM_EXAMPLES_TO_PRINT:
            print(f"\n\n==================== 评测样本 {i+1} ====================")
            print(f" [查询 Query]: {query}")
            gt_names = [t.get('name', 'N/A') for t in ground_truth]
            print(f" [真实工具 Ground Truth]: {gt_names}")
            retrieved_names = [t.get('name', 'N/A') for t in retrieved_tools]
            print(f" [召回结果 Prediction @{max_k}]: {retrieved_names}")
            is_complete = calculate_completeness_at_k(retrieved_tools, ground_truth, k=max_k)
            status = "✅ 完整命中 (Complete Hit)" if is_complete else "❌ 部分或偏离 (Partial/Miss)"
            print(f" [评测状态 @{max_k}]: {status}")
            print("=====================================================")

        # 计算指标的逻辑不变
        for k in K_VALUES:
            results['Recall@K'][k].append(calculate_recall_at_k(retrieved_tools, ground_truth, k))
            results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved_tools, ground_truth, k))
            results['COMP@K'][k].append(calculate_completeness_at_k(retrieved_tools, ground_truth, k))
            
    # --- 5. 汇总并报告结果 ---
    print("\n\n--- 步骤 5: 评测结果报告 ---")
    final_scores = {metric: {k: np.mean(scores) for k, scores in k_scores.items()} for metric, k_scores in results.items()}
    report_df = pd.DataFrame(final_scores).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    
    print("【纯净描述版】BM25 召回模型评测结果:")
    print("-" * 40)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print("-" * 40)

if __name__ == "__main__":
    main()