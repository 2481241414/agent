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
from sklearn.metrics import roc_auc_score

# --- 模型库 ---
from rank_bm25 import BM25Okapi
import jieba
import faiss
import torch
# ★★★ 新增 CrossEncoder 用于 Reranker ★★★
from sentence_transformers import SentenceTransformer, CrossEncoder


# --- 召回器 1: BM25 ---
class BM25ToolRetriever:
    # --- 内容完全不变 ---
    def __init__(self, all_tools_corpus: list, all_tools_definitions: list, k1=1.5, b=0.75):
        self.tool_corpus = all_tools_corpus
        self.tool_definitions = all_tools_definitions
        self.k1 = k1
        self.b = b
        self._add_jieba_words()
        tokenized_corpus = [self._tokenize(doc) for doc in self.tool_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
    def _tokenize(self, text: str) -> list[str]:
        return jieba.lcut(text, cut_all=False)
    def _add_jieba_words(self):
        for tool in self.tool_definitions:
            jieba.add_word(tool.get('name', ''), freq=100)
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "降价", "签到", "积分", "发票", "开票", "报销凭证"]
        for word in core_words:
            jieba.add_word(word, freq=100)
    def retrieve_with_scores(self, query: str, top_k: int):
        tokenized_query = self._tokenize(query)
        all_scores = self.bm25.get_scores(tokenized_query)
        # 使用 np.argsort 获取索引
        top_k_indices = np.argsort(all_scores)[-top_k:][::-1]
        retrieved = [self.tool_definitions[i] for i in top_k_indices if all_scores[i] > 0]
        return retrieved, all_scores


# --- 召回器 2: Faiss + BGE-M3 ---
class FaissToolRetriever:
    # --- 内容完全不变 ---
    def __init__(self, all_tools_corpus: list, all_tools_definitions: list):
        print("--- 正在初始化 FaissToolRetriever... ---")
        self.tool_definitions = all_tools_definitions
        self.corpus = all_tools_corpus
        print("1. 加载 BAAI/bge-m3 embedding 模型...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('/home/workspace/lgq/shop/model/bge-m3', device=self.device)
        print(f"模型已加载到 {self.device} 设备。")
        print("2. 正在将工具语料库编码为向量...")
        self.corpus_embeddings = self.model.encode(self.corpus, normalize_embeddings=True, show_progress_bar=True, batch_size=32)
        print("3. 正在构建 FAISS 索引...")
        embedding_dim = self.corpus_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        self.index = faiss.IndexIDMap(index)
        doc_ids = np.arange(len(self.corpus))
        self.index.add_with_ids(self.corpus_embeddings, doc_ids)
        print(f"FAISS 索引构建完成！共 {self.index.ntotal} 个向量。\n")
    def retrieve_with_scores(self, query: str, top_k: int):
        instruction = "Represent this sentence for searching relevant passages: "
        query_embedding = self.model.encode([instruction + query], normalize_embeddings=True, show_progress_bar=False)[0]
        all_scores = query_embedding @ self.corpus_embeddings.T
        distances, indices = self.index.search(np.expand_dims(query_embedding, axis=0), top_k)
        valid_indices = indices[0][indices[0] != -1]
        retrieved = [self.tool_definitions[i] for i in valid_indices]
        return retrieved, all_scores


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ 新增：两阶段精排召回器 (BM25召回 + BGE-Reranker精排) ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
class RerankRetriever:
    def __init__(self, first_stage_retriever, reranker_model_path='BAAI/bge-reranker-v2-m3'):
        print("--- 正在初始化 RerankRetriever... ---")
        
        # 1. 持有一个已经初始化好的第一阶段召回器
        self.first_stage_retriever = first_stage_retriever
        self.tool_definitions = self.first_stage_retriever.tool_definitions
        
        # 2. 加载 Reranker 模型 (CrossEncoder)
        print(f"1. 加载 {reranker_model_path} Reranker 模型...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 您也可以将 'bge-reranker-base' 下载到本地，然后加载本地路径
        self.reranker = CrossEncoder(reranker_model_path, max_length=512, device=self.device)
        print(f"Reranker 模型已加载到 {self.device} 设备。\n")
        
        # 3. 定义第一阶段召回的数量，这个值可以调整
        self.recall_k = 5

    def retrieve_with_scores(self, query: str, top_k: int):
        # --- Stage 1: Recall (使用 BM25) ---
        initial_candidates, _ = self.first_stage_retriever.retrieve_with_scores(query, top_k=self.recall_k)
        
        if not initial_candidates:
            return [], np.array([])

        # --- Stage 2: Rerank ---
        # 准备 reranker 的输入：[(query, doc_text), (query, doc_text), ...]
        # 从 first_stage_retriever 获取候选工具对应的语料文本
        candidate_corpus = []
        for tool in initial_candidates:
            # 找到工具定义在列表中的索引，从而在语料库中找到对应的文档
            doc_index = self.tool_definitions.index(tool)
            candidate_corpus.append(self.first_stage_retriever.tool_corpus[doc_index])

        rerank_input = [[query, doc] for doc in candidate_corpus]
        
        # 使用 reranker 模型进行打分
        rerank_scores = self.reranker.predict(rerank_input, show_progress_bar=False)
        
        # 将 rerank 分数与候选工具配对
        scored_candidates = list(zip(rerank_scores, initial_candidates))
        
        # 按 rerank 分数降序排序
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 提取排序后的工具
        reranked_tools = [candidate for score, candidate in scored_candidates]
        
        # 为了评测AUC，我们需要一个包含所有工具分数的数组。
        # 我们给rerank过的工具赋予它们的分数，其他的工具给一个极低的分数。
        all_scores = np.full(len(self.tool_definitions), -999.0, dtype=np.float32)
        for score, tool in scored_candidates:
            idx = self.tool_definitions.index(tool)
            all_scores[idx] = score

        return reranked_tools[:top_k], all_scores


# --- (评测函数和辅助函数保持不变) ---
def _get_tool_names(tools: list) -> set:
    # ...内容不变...
    if not isinstance(tools, list): return set()
    return {tool.get('name') for tool in tools}

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...内容不变...
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    return len(retrieved_names_at_k.intersection(ground_truth_names)) / len(ground_truth_names)

def calculate_completeness_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...内容不变...
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    return 1.0 if ground_truth_names.issubset(retrieved_names_at_k) else 0.0

def calculate_ndcg_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...内容不变...
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, tool in enumerate(retrieved[:k]) if tool.get('name') in ground_truth_names)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth_names), k)))
    return dcg / idcg if idcg > 0 else 0.0

def calculate_hit_ratio_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...内容不变...
    retrieved_names = _get_tool_names(retrieved[:k])
    return 1.0 if retrieved_names & _get_tool_names(ground_truth) else 0.0

def calculate_average_precision_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    # ...内容不变...
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
    # ...内容不变...
    gt_names = _get_tool_names(ground_truth)
    for i, tool in enumerate(retrieved[:k]):
        if tool.get('name') in gt_names:
            return 1.0 / (i + 1)
    return 0.0

def calculate_auc_for_query(all_scores: np.ndarray, tool_defs: list, ground_truth: list) -> float:
    # ...内容不变...
    gt_names = _get_tool_names(ground_truth)
    if not gt_names: return 0.5
    labels = [1 if t['name'] in gt_names else 0 for t in tool_defs]
    try:
        if len(set(labels)) < 2: return 0.5
        return roc_auc_score(labels, all_scores)
    except ValueError: return 0.5

def get_exact_tool_definitions():
    # ...内容不变...
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

def get_synonyms():
    # ...内容不变...
    return { "购物车": ["采购车"], "收藏": ["喜欢", "想要", "关注", "收藏夹"], "找订单": ["查订单", "搜订单"], "看物流": ["查快递", "包裹进度"], "买的东西": ["我买的", "收到的", "购买记录"], "卖的东西": ["我卖的", "发出的", "售出记录"], "发票": ["开票", "报销凭证"], }

def build_corpus(data_df: pd.DataFrame, tool_definitions: list) -> list:
    # ...内容不变...
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

def run_evaluation(retriever, retriever_name: str, test_df: pd.DataFrame, all_tools_definitions: list):
    # --- 内容完全不变 ---
    K_VALUES = [1, 2, 3, 4, 5]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10
    print(f"\n{'='*20} 开始评测: {retriever_name} {'='*20}")
    results = { 'Recall@K': {k: [] for k in K_VALUES}, 'HR@K': {k: [] for k in K_VALUES}, 'MAP@K': {k: [] for k in K_VALUES}, 'MRR@K': {k: [] for k in K_VALUES}, 'NDCG@K': {k: [] for k in K_VALUES}, 'COMP@K': {k: [] for k in K_VALUES}, 'AUC': [], 'processing_time': [] }
    error_cases = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"评测 {retriever_name}"):
        query = row['final_query']
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
            error_cases.append({ "Query": query, "Ground Truth": gt_name, "Prediction@1": pred_name_top1, "Prediction@5": [r.get('name') for r in retrieved] })
    print(f"\n--- {retriever_name} 评测结果报告 ---")
    final_scores = {}
    for metric, vals in results.items():
        if metric == 'AUC': final_scores['AUC'] = np.mean(vals)
        elif metric == 'processing_time': continue
        else: final_scores[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    report_df = pd.DataFrame({ m: final_scores[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    print("-" * 50)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores['AUC']:.4f}")
    print("-" * 50)
    total_time, avg_time_ms = np.sum(results['processing_time']), np.mean(results['processing_time']) * 1000
    qps = len(test_df) / total_time if total_time > 0 else 0
    print("\n性能评测:")
    print(f"总耗时: {total_time:.4f} 秒, 平均每条耗时: {avg_time_ms:.4f} 毫秒, QPS: {qps:.2f}")
    print("-" * 50)
    print(f"\n--- {retriever_name} Top-1 错误案例分析 (共 {len(error_cases)} 个错误) ---")
    for i, case in enumerate(error_cases[:NUM_ERROR_EXAMPLES_TO_PRINT]):
        print(f"\n--- 错误案例 {i+1}/{len(error_cases)} ---")
        print(f"  [查询 Query]: {case['Query']}")
        print(f"  [真实工具 Ground Truth]: {case['Ground Truth']}")
        print(f"  [预测工具 Prediction@1]: {case['Prediction@1']}")
        print(f"  [预测工具 Prediction@5]: {case['Prediction@5']}")
    print("-" * 50)


# --- 主程序 ---
def main():
    # --- 0. 配置区域 ---
    annotated_data_file_path = r'/home/workspace/lgq/shop/data/corrected_data.csv' 
    
    # --- 1. 数据加载与划分 ---
    print("--- 步骤 1: 加载已标注数据并划分 ---")
    try:
        data_df = pd.read_csv(annotated_data_file_path)
    except Exception as e:
        print(f"错误: 读取文件 '{annotated_data_file_path}' 失败. {e}")
        return
    
    def parse_tools(tool_string):
        try: return ast.literal_eval(tool_string)
        except (ValueError, SyntaxError): return []
    data_df['available_tools'] = data_df['available_tools'].apply(parse_tools)
    
    train_df = data_df[data_df['is_train'] == 0].copy()
    test_df = data_df[data_df['is_train'] == 1].copy()
    print(f"数据加载完成：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条。\n")

    # --- 2. 构建通用语料库和工具定义 ---
    all_tools_definitions = get_exact_tool_definitions()
    final_corpus = build_corpus(train_df, all_tools_definitions)
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 3. 初始化并评测三个召回器：BM25 vs FAISS+BGE vs BM25+Reranker ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    
    # (A) 运行 BM25 评测 (基线)
    # ------------------------------------------------
    # print("开始构建 BM25 召回器 (作为基线)...")
    bm25_retriever = BM25ToolRetriever(final_corpus, all_tools_definitions, k1=1.2, b=0.75)
    # print("BM25 召回器构建完成。")
    # run_evaluation(bm25_retriever, "BM25", test_df, all_tools_definitions)
    
    
    # (B) 运行 Faiss + BGE-M3 评测
    # ------------------------------------------------
    # print("\n\n开始构建 Faiss + BGE-M3 召回器...")
    # faiss_retriever = FaissToolRetriever(final_corpus, all_tools_definitions)
    # run_evaluation(faiss_retriever, "Faiss+BGE-M3", test_df, all_tools_definitions)

    # ★★★ (C) 运行 BM25 + Reranker 评测 (两阶段精排) ★★★
    # ------------------------------------------------
    rerank_retriever = RerankRetriever(first_stage_retriever=bm25_retriever, reranker_model_path='/home/workspace/lgq/shop/model/bge-reranker-v2-m3')
    run_evaluation(rerank_retriever, "BM25 + BGE-Reranker", test_df, all_tools_definitions)


if __name__ == "__main__":
    main()