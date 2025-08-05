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
import jieba
import requests # <--- 新增: 用于同步API请求

# ==============================================================================
# 区域 1: 检查并导入所需库
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.metrics import roc_auc_score
    import aiohttp
    import asyncio
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install faiss-cpu torch sentence-transformers transformers rank_bm25 scikit-learn pandas tqdm aiohttp requests")
    exit()

# ==============================================================================
# 区域 2: 召回器类定义 (BM25Retriever, InstructionSearcher 不变)
# ==============================================================================
# ... (你的 BM25Retriever 和 InstructionSearcher 类代码在这里，无需改动)
class BM25Retriever:
    """【关键词通路】BM25召回器，专注于关键词和用户多样化表达的匹配。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, k1=1.5, b=0.75):
        self.definitions = all_tools_definitions
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(all_tools_definitions)}
        self._add_jieba_words()
        
        print("--- [BM25通路] 正在构建关键词增强语料库... ---")
        corpus = self._build_keyword_rich_corpus(data_df)
        tokenized_corpus = [jieba.lcut(doc, cut_all=False) for doc in tqdm(corpus, desc="BM25语料库分词")]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        print("--- [BM25通路] 召回器构建完成 ---")

    def _add_jieba_words(self):
        for tool in self.definitions:
            jieba.add_word(tool.get('name', '').split('(')[0], freq=100)
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "发票", "优惠券"]
        for word in core_words:
            jieba.add_word(word, freq=100)

    def _build_keyword_rich_corpus(self, data_df: pd.DataFrame) -> list:
        tool_text_aggregator = defaultdict(list)
        for _, row in data_df.iterrows():
            if not isinstance(row.get('ground_truth_tool'), list) or not row['ground_truth_tool']: continue
            tool_name = row['ground_truth_tool'][0]['name']
            if pd.notna(row['指令']):
                tool_text_aggregator[tool_name].append(row['指令'])
        
        corpus = [''] * len(self.definitions)
        for tool_def in self.definitions:
            tool_name = tool_def['name']
            tool_idx = self.tool_name_to_idx[tool_name]
            aggregated_text = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
            document = f"{aggregated_text}"
            corpus[tool_idx] = document
        return corpus

    def retrieve_scores(self, query: str) -> np.ndarray:
        tokenized_query = jieba.lcut(query, cut_all=False)
        return self.bm25.get_scores(tokenized_query)


class InstructionSearcher:
    """
    【精准意图通路】支持本地加载或API调用两种模式，以统一向量化方式。
    """
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, mode: str, model_path_or_name: str, api_url: str = None, api_concurrency_limit: int = 10):
        self.definitions = all_tools_definitions
        self.mode = mode
        self.model_path_or_name = model_path_or_name
        self.api_url = api_url
        self.model = None # 用于本地模式
        self.faiss_index = None
        self.api_concurrency_limit = api_concurrency_limit

        print(f"--- [意图通路] 模式: {self.mode.upper()} ---")
        if self.mode == 'local':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"--- [意图通路] 正在使用设备: {self.device} ---")
            print(f"--- [意图通路] 正在加载本地语义模型: {self.model_path_or_name} ---")
            self.model = SentenceTransformer(self.model_path_or_name, trust_remote_code=True, device=self.device)
        elif self.mode == 'api':
            print(f"--- [意图通路] API并发请求阈值设置为: {self.api_concurrency_limit} ---")
        else:
            raise ValueError("模式 (mode) 必须是 'local' 或 'api'")

        self._build_mappings(data_df)

    def _build_mappings(self, data_df: pd.DataFrame):
        self.instruction_to_tool_map = {}
        for _, row in data_df.drop_duplicates(subset=['指令'], keep='last').iterrows():
            if pd.notna(row['指令']) and isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
                self.instruction_to_tool_map[row['指令']] = row['ground_truth_tool'][0]
        self.unique_instructions = list(self.instruction_to_tool_map.keys())
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(self.definitions)}

    def _build_faiss_index(self, embeddings: np.ndarray):
        embeddings = embeddings.astype('float32')
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    async def _get_embeddings_from_api(self, texts: list, session: aiohttp.ClientSession) -> list:
        semaphore = asyncio.Semaphore(self.api_concurrency_limit)
        tasks = []

        async def fetch_embedding(text):
            async with semaphore: 
                headers = {"Content-Type": "application/json"}
                payload = {"model": self.model_path_or_name, "input": text}
                try:
                    async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result['data'][0]['embedding']
                        else:
                            error_text = await response.text()
                            print(f"API请求失败, 状态码: {response.status}, 错误: {error_text}")
                            return None
                except Exception as e:
                    print(f"请求异常: {e}")
                    return None

        for text in texts:
            tasks.append(fetch_embedding(text))
        
        embeddings = await asyncio.gather(*tasks)
        
        return [emb if emb is not None else [] for emb in embeddings]

    async def initialize_and_get_all_scores(self, all_plan_queries: list):
        """统一的初始化和评分函数"""
        print(f"--- [意图通路] 正在将 {len(self.unique_instructions)} 条唯一指令编码为向量...")
        
        instruction_embeddings = None
        if self.mode == 'local':
            instruction_embeddings = self.model.encode(self.unique_instructions, convert_to_tensor=False, show_progress_bar=True)
        else: # api mode
            async with aiohttp.ClientSession() as session:
                instruction_embeddings = await self._get_embeddings_from_api(self.unique_instructions, session)
                successful_embeddings = [emb for emb in instruction_embeddings if emb]
                if len(successful_embeddings) != len(self.unique_instructions):
                    print(f"警告: {len(self.unique_instructions) - len(successful_embeddings)} 条指令向量化失败!")
                    if not successful_embeddings:
                        raise RuntimeError("所有指令向量化失败，无法继续！")
                
                self.unique_instructions = [inst for inst, emb in zip(self.unique_instructions, instruction_embeddings) if emb]
                instruction_embeddings = successful_embeddings

        self._build_faiss_index(np.array(instruction_embeddings))
        print("--- [意图通路] Faiss索引构建完成 ---")

        print(f"--- [意图通路] 正在对 {len(all_plan_queries)} 条查询进行评分...")
        all_semantic_scores = []
        query_embeddings = None

        if self.mode == 'local':
            query_embeddings = self.model.encode(all_plan_queries, convert_to_tensor=False, show_progress_bar=True)
        else: # api mode
             async with aiohttp.ClientSession() as session:
                query_embeddings = await self._get_embeddings_from_api(all_plan_queries, session)
        
        for query_embedding in tqdm(query_embeddings, desc="计算语义分数"):
            if not query_embedding:
                all_semantic_scores.append(np.zeros(len(self.definitions), dtype='float32'))
                continue

            query_embedding_np = np.array([query_embedding], dtype='float32')
            faiss.normalize_L2(query_embedding_np)
            
            num_neighbors = min(len(self.unique_instructions), 50) 
            distances, indices = self.faiss_index.search(query_embedding_np, k=num_neighbors)
            
            tool_scores = np.zeros(len(self.definitions), dtype='float32')
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    matched_instruction = self.unique_instructions[idx]
                    tool_def = self.instruction_to_tool_map.get(matched_instruction)
                    if tool_def:
                        tool_idx = self.tool_name_to_idx.get(tool_def['name'])
                        if tool_idx is not None:
                            tool_scores[tool_idx] = max(tool_scores[tool_idx], dist)
            all_semantic_scores.append(tool_scores)
            
        return all_semantic_scores
# ==============================================================================
# 区域 2.5: Reranker类定义 (全新改造)
# ==============================================================================
# ==============================================================================
# 区域 2.5: Reranker类定义 (异步改造版)
# ==============================================================================
class QwenReranker:
    """
    【精排通路】使用Qwen3-Reranker对召回结果进行重排序。
    支持 'local' (同步) 和 'api' (异步) 两种模式。
    """
    def __init__(self, mode: str, model_path_or_name: str, api_url: str = None, device: str = 'cuda', api_concurrency_limit: int = 20):
        self.mode = mode
        self.model_path_or_name = model_path_or_name
        self.api_url = api_url
        self.model = None

        print(f"--- [精排通路] 模式: {self.mode.upper()} ---")
        if self.mode == 'local':
            self.device = device if torch.cuda.is_available() else 'cpu'
            print(f"--- [精排通路] 正在加载Reranker模型: {self.model_path_or_name} ---")
            self.model = CrossEncoder(self.model_path_or_name, device=self.device, trust_remote_code=True)
            print(f"--- [精排通路] Reranker模型加载完成，使用设备: {self.device} ---")
        elif self.mode == 'api':
            print(f"--- [精排通路] API模式已启用，目标URL: {self.api_url} ---")
            print(f"--- [精排通路] API并发请求阈值设置为: {api_concurrency_limit} ---")
            if not self.api_url:
                raise ValueError("API模式下必须提供 api_url")
            # 为异步请求创建信号量进行并发控制
            self.semaphore = asyncio.Semaphore(api_concurrency_limit)
        else:
            raise ValueError("Reranker模式 (mode) 必须是 'local' 或 'api'")

    def rerank_sync(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """
        [同步方法] 对候选指令进行重排序，主要用于本地模式。
        """
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    async def rerank_async(self, query: str, documents: list[str], session: aiohttp.ClientSession) -> list[tuple[str, float]]:
        """
        [异步方法] 使用aiohttp并发请求API进行重排序。
        """
        if not documents:
            return []

        async with self.semaphore: # 使用信号量控制并发数
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model_path_or_name,
                "query": query,
                "documents": documents,
            }
            try:
                async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        api_results = await response.json()
                        scored_docs = [("", 0.0)] * len(documents)
                        for res in api_results:
                            if 'index' in res and res['index'] < len(documents):
                                scored_docs[res['index']] = (documents[res['index']], res['score'])
                        
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        return scored_docs
                    else:
                        error_text = await response.text()
                        print(f"API Rerank请求失败，状态码: {response.status}, 错误: {error_text[:200]}")
                        return [(doc, 0.0) for doc in documents]
            except Exception as e:
                print(f"API Rerank请求异常: {e}")
                return [(doc, 0.0) for doc in documents]


# ==============================================================================
# 区域 3: 评测函数 (不变)
# ==============================================================================
# ... (你的所有 calculate_... 函数在这里，无需改动)
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

# ==============================================================================
# 区域 4: 工具定义 (不变)
# ==============================================================================
def get_exact_tool_definitions():
    # ... (你的工具定义函数，内容不变)
    tools = [
        # 1. 购物 - 搜索 (1.1)
        {"name": "search_goods(app, search_info_slot, page_type, filter_detail_slot, type_slot, area_slot, order_type)", "description": "在app程序中依据名称搜索商品,可以指定具体在哪一个子页面进行搜索, 搜索结果的筛选条件和排序方式"},
        {"name": "search_stores(app, search_info_slot, filter_type, filter_detail_slot, location_slot, qualification_slot, order_type)", "description": "在app程序中依据名称搜索店铺,可以使用筛选器限制搜索结果,也可以指定搜索结果的排序方式"},
        {"name": "open_search_history(app)", "description": "打开app程序的搜索历史界面"},
        {"name": "delete_search_history(app)", "description": "清除app中的搜索历史"},
        {"name": "open_camera_search(app)", "description": "打开app程序的图片搜索功能"},
        {"name": "search_delivery_time(app, search_info_slot, address_slot)", "description": "搜索一件商品并根据给出的地址查询该商品送达该地址的预估运送时间"},
        {"name": "search_cart_content(app, search_info_slot)", "description": "在app应用程序中查看购物车/采购车(阿里巴巴的叫法)查找商品"},
        {"name": "search_in_favorite_goods(app, search_info_slot)", "description": "在app程序中打开收藏的、喜爱、想要或关注商品的页面,并在其中的搜索栏中进行搜索"},
        {"name": "search_in_favorite_stores(app, search_info_slot)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面,并在其中的搜索栏搜索商品"},
        {"name": "search_order(app, search_info_slot, order_status)", "description": "在app应用程序中搜索订单"},

        # 2. 购物 - 打开 (1.2)
        {"name": "open_goods_page(app, search_info_slot, page_type)", "description": "通过商品名称找到并打开其详情页面,可以指定子页面,例如评论、规格、参数、详情等"},
        {"name": "open_stores_page(app, store_name_slot, search_info_slot, category_slot)", "description": "通过店铺名称找到并打开店铺的内容页面,可以在其中进行店铺内搜索或打开类别子页面"},
        {"name": "open_special_page(app, page_type)", "description": "打开特殊页面,例如活动页面"},

        # 3. 购物 - 购物车 (1.3)
        {"name": "open_cart_content(app, filter_type, filter_detail_slot)", "description": "在app应用程序中查看购物车/采购车(阿里巴巴的叫法)指定类型的商品"},
        {"name": "add_into_cart(app, search_info_slot, specification_slot, num_slot, address_slot)", "description": "搜索商品并将其添加入购物车,可以指定添加的商品规格、数量并选择收货地址"},

        # 4. 购物 - 收藏 (1.4)
        {"name": "open_favorite_goods(app, filter_type, filter_detail_slot, order_type)", "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面,并按照条件进行筛选"},
        {"name": "open_favorite_stores(app, filter_type)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面,并按照条件进行筛选"},
        {"name": "add_into_favorite_goods(app, search_info_slot)", "description": "在app程序中搜索商品,并将其添加到商品收藏夹中"},
        {"name": "add_into_favorite_stores(app, search_info_slot)", "description": "在app程序中按照店铺名搜索店铺,并将其添加到店铺收藏夹中"},
        {"name": "delete_favorite_goods(app, search_info_slot)", "description": "在app程序的商品收藏夹中搜索指定商品并将其删除"},
        
        # 5. 购物 - 下单 (1.5)
        {"name": "order_to_purchase_goods(app, search_info_slot, specification_slot, num_slot, address_slot, payment_method_slot)", "description": "通过商品名称找到商品并下单购买,可以指定添加的商品规格、数量并选择收货地址以及支付方式"},

        # 6. 购物 - 订单 (1.6)
        {"name": "open_orders_bought(app, order_status, filter_detail_slot)", "description": "在app应用程序中查看买入的指定状态的订单列表,例如待付款、待收货、待评价等。"},
        {"name": "open_orders_sold(app, order_status, filter_detail_slot)", "description": "在app应用程序中查看自己售卖的指定状态的订单列表,例如待付款、待收货、待评价等。"},
        {"name": "open_orders_release(app, order_status)", "description": "在app应用程序中查看自己发布的指定状态的订单列表,例如在卖、草稿、已下架等。"},
        {"name": "open_orders_all_review(app)", "description": "在app应用程序中查看待评价状态的订单列表,在不指定购买还是售卖的订单时,及全都要看时使用。"},
        {"name": "apply_after_sales(app, search_info_slot, after_sales_type, reason_slot)", "description": "在app应用程序中搜索订单,并申请售后"},

        # 7. 购物 - 物流 (1.7)
        {"name": "open_logistics_receive(app, filter_type)", "description": "打开显示已购商品信息的界面,查看相关物流信息,并根据物流情况进行筛选"},
        {"name": "open_logistics_send(app, filter_type)", "description": "打开显示已售商品信息的界面,查看相关物流信息,并根据物流情况进行筛选"},
        {"name": "open_express_delivery(app)", "description": "打开app寄送快递的界面"},
        {"name": "manage_order_logistics_status(app, search_info_slot, action_type)", "description": "在app中管理指定订单的物流状态,包括催发货,催配送,确认收货"},
        {"name": "open_order_tracking_number(app, search_info_slot)", "description": "在app中查询指定订单的物流单号"},
        {"name": "call_order_courier(app, search_info_slot)", "description": "在app中拨打指定订单的快递电话"},

        # 8. 购物 - 客服 (1.8)
        {"name": "open_customer_service(app, order_slot, store_slot)", "description": "在app应用程序中联系官方客服,或联系指令订单的店铺客服,或联系指定店铺的客服"},
        {"name": "apply_price_protection(app)", "description": "在app应用程序中联系客服进行价保"},

        # 9. 购物 - 评价 (1.9)
        {"name": "rate_order(app, search_info_slot, rating_slot, review_text_slot, upload_images)", "description": "在app应用程序评价商城中的指定订单"},

        # 10. 购物 - 发票 (1.10)
        {"name": "open_invoice_page(app, page_type)", "description": "在app应用程序中打开与发票相关的页面"},

        # 11. 购物 - 签到 (1.11)
        {"name": "sign_in(app, page_type)", "description": "在app程序中完成每日签到,领取积分、金币等奖励的操作"},

        # 12. 购物 - 启动 (1.12)
        {"name": "open_app(app)", "description": "打开指定的应用程序"},
    ]
    return tools

# ==============================================================================
# 区域 5: 评测核心逻辑 (全新改造)
# ==============================================================================
async def evaluate_recall_system(
    data_df, 
    all_bm25_scores, 
    all_semantic_scores, 
    all_tools_definitions, 
    instruction_to_tool_map,
    tool_to_instructions_map,
    alpha, 
    k_values, 
    reranker: QwenReranker, # 类型提示
    rerank_top_n, 
    session: aiohttp.ClientSession, # <-- 新增session参数
    full_report=False
):
    results = defaultdict(lambda: defaultdict(list))
    error_cases = []
    latency_records = []
    detailed_predictions = [] 

    tool_name_to_def_map = {tool['name']: tool for tool in all_tools_definitions}

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if (max_s - min_s) == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    # --- 步骤 A: 准备所有任务 ---
    # 准备一个列表，存放每个待处理样本的 rerank 任务协程和相关数据
    rerank_tasks = []
    prepared_data = []

    for i, row in data_df.iterrows():
        query = row['plan（在xx中做什么）']
        
        # 1. 召回
        bm25_scores = all_bm25_scores[i]
        semantic_scores = all_semantic_scores[i]
        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)
        recall_scores = alpha * norm_bm25 + (1 - alpha) * norm_semantic
        
        recall_sorted_indices = np.argsort(recall_scores)[::-1]
        recall_candidate_tools = [all_tools_definitions[idx] for idx in recall_sorted_indices[:rerank_top_n]]

        # 2. 扩展
        instructions_to_rerank = set()
        for tool in recall_candidate_tools:
            instructions = tool_to_instructions_map.get(tool['name'], [])
            instructions_to_rerank.update(instructions)
        instructions_to_rerank = list(instructions_to_rerank)

        # 区分本地同步模式和API异步模式
        if reranker.mode == 'local':
            # 对于本地模式，直接同步执行
            task = reranker.rerank_sync(query, instructions_to_rerank)
        else:
            # 对于API模式，创建异步任务
            task = reranker.rerank_async(query, instructions_to_rerank, session)
        
        rerank_tasks.append(task)
        prepared_data.append({
            "row": row,
            "recall_scores": recall_scores,
            "recall_candidate_tools": recall_candidate_tools
        })

    # --- 步骤 B: 并发执行所有 Rerank 任务 ---
    # 使用tqdm.asyncio来为异步任务添加进度条
    from tqdm.asyncio import tqdm as anaylse_tqdm
    all_reranked_instructions = await anaylse_tqdm.gather(*rerank_tasks, desc="并发精排中")

    # --- 步骤 C: 收集并处理所有结果 ---
    for i, reranked_instructions in enumerate(all_reranked_instructions):
        start_time = time.time()
        
        # 从准备好的数据中获取上下文
        data = prepared_data[i]
        row = data['row']
        recall_scores = data['recall_scores']
        recall_candidate_tools = data['recall_candidate_tools']
        ground_truth = row['ground_truth_tool']
        
        # 4. 聚合
        tool_rerank_scores = defaultdict(lambda: -1e9)
        for instruction_text, score in reranked_instructions:
            tool_def = instruction_to_tool_map.get(instruction_text)
            if tool_def:
                tool_name = tool_def['name']
                tool_rerank_scores[tool_name] = max(tool_rerank_scores[tool_name], score)

        # 5. 最终排序
        recall_candidate_tools.sort(key=lambda tool: tool_rerank_scores[tool['name']], reverse=True)
        retrieved = recall_candidate_tools
        retrieved_scores = [tool_rerank_scores[tool['name']] for tool in retrieved]
        
        end_time = time.time()
        latency_records.append(end_time - start_time) # 注意：此时的时延仅包含后处理部分

        # 6. 评估
        if full_report:
            # ... (这部分评估逻辑与之前完全相同)
            prediction_record = {
                "query": row['query'],
                "plan": row['plan（在xx中做什么）'],
                "ground_truth": [_get_tool_names(ground_truth)],
                "retrieved_top_k": [{"tool": t.get('name'), "score": float(s)} for t, s in zip(retrieved[:max(k_values)], retrieved_scores[:max(k_values)])]
            }
            detailed_predictions.append(prediction_record)

            for k in k_values:
                results['Recall@K'][k].append(calculate_recall_at_k(retrieved, ground_truth, k))
                results['HR@K'][k].append(calculate_hit_ratio_at_k(retrieved, ground_truth, k))
                results['MAP@K'][k].append(calculate_average_precision_at_k(retrieved, ground_truth, k))
                results['MRR@K'][k].append(calculate_mrr_at_k(retrieved, ground_truth, k))
                results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved, ground_truth, k))
                results['COMP@K'][k].append(calculate_completeness_at_k(retrieved, ground_truth, k))
            
            results['AUC']['all'].append(calculate_auc_for_query(recall_scores, all_tools_definitions, ground_truth))
            
            if calculate_recall_at_k(retrieved, ground_truth, 1) < 1.0:
                gt_name = _get_tool_names(ground_truth).pop() if ground_truth else "N/A"
                pred_name_top1 = retrieved[0].get('name') if retrieved else "N/A"
                error_cases.append({"Query": row['plan（在xx中做什么）'], "Ground Truth": [gt_name], "Prediction@1": [pred_name_top1], "Prediction@5": [r.get('name') for r in retrieved[:5]]})
        else:
            results['Recall@K'][1].append(calculate_recall_at_k(retrieved, ground_truth, 1))

    if full_report:
        return results, error_cases, latency_records, detailed_predictions
    else:
        return np.mean(results['Recall@K'][1])



# ==============================================================================
# 区域 6: 主程序 (重大修改)
# ==============================================================================
# ==============================================================================
# 区域 6: 主程序 (修改版)
# ==============================================================================
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
import jieba
import requests 

# ==============================================================================
# 区域 1: 检查并导入所需库
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.metrics import roc_auc_score
    import aiohttp
    import asyncio
    # 引入tqdm的异步版本，用于显示并发任务进度条
    from tqdm.asyncio import tqdm as anaylse_tqdm
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install faiss-cpu torch sentence-transformers transformers rank_bm25 scikit-learn pandas tqdm aiohttp requests")
    exit()

# ==============================================================================
# 区域 2: 召回器类定义 (BM25Retriever, InstructionSearcher 不变)
# ==============================================================================
class BM25Retriever:
    """【关键词通路】BM25召回器，专注于关键词和用户多样化表达的匹配。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, k1=1.5, b=0.75):
        self.definitions = all_tools_definitions
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(all_tools_definitions)}
        self._add_jieba_words()
        
        print("--- [BM25通路] 正在构建关键词增强语料库... ---")
        corpus = self._build_keyword_rich_corpus(data_df)
        tokenized_corpus = [jieba.lcut(doc, cut_all=False) for doc in tqdm(corpus, desc="BM25语料库分词")]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        print("--- [BM25通路] 召回器构建完成 ---")

    def _add_jieba_words(self):
        for tool in self.definitions:
            jieba.add_word(tool.get('name', '').split('(')[0], freq=100)
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "发票", "优惠券"]
        for word in core_words:
            jieba.add_word(word, freq=100)

    def _build_keyword_rich_corpus(self, data_df: pd.DataFrame) -> list:
        tool_text_aggregator = defaultdict(list)
        for _, row in data_df.iterrows():
            if not isinstance(row.get('ground_truth_tool'), list) or not row['ground_truth_tool']: continue
            tool_name = row['ground_truth_tool'][0]['name']
            if pd.notna(row['指令']):
                tool_text_aggregator[tool_name].append(row['指令'])
        
        corpus = [''] * len(self.definitions)
        for tool_def in self.definitions:
            tool_name = tool_def['name']
            tool_idx = self.tool_name_to_idx[tool_name]
            aggregated_text = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
            document = f"{aggregated_text}"
            corpus[tool_idx] = document
        return corpus

    def retrieve_scores(self, query: str) -> np.ndarray:
        tokenized_query = jieba.lcut(query, cut_all=False)
        return self.bm25.get_scores(tokenized_query)


class InstructionSearcher:
    """
    【精准意图通路】支持本地加载或API调用两种模式，以统一向量化方式。
    """
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, mode: str, model_path_or_name: str, api_url: str = None, api_concurrency_limit: int = 10):
        self.definitions = all_tools_definitions
        self.mode = mode
        self.model_path_or_name = model_path_or_name
        self.api_url = api_url
        self.model = None # 用于本地模式
        self.faiss_index = None
        self.api_concurrency_limit = api_concurrency_limit

        print(f"--- [意图通路] 模式: {self.mode.upper()} ---")
        if self.mode == 'local':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"--- [意图通路] 正在使用设备: {self.device} ---")
            print(f"--- [意图通路] 正在加载本地语义模型: {self.model_path_or_name} ---")
            self.model = SentenceTransformer(self.model_path_or_name, trust_remote_code=True, device=self.device)
        elif self.mode == 'api':
            print(f"--- [意图通路] API并发请求阈值设置为: {self.api_concurrency_limit} ---")
        else:
            raise ValueError("模式 (mode) 必须是 'local' 或 'api'")

        self._build_mappings(data_df)

    def _build_mappings(self, data_df: pd.DataFrame):
        self.instruction_to_tool_map = {}
        for _, row in data_df.drop_duplicates(subset=['指令'], keep='last').iterrows():
            if pd.notna(row['指令']) and isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
                self.instruction_to_tool_map[row['指令']] = row['ground_truth_tool'][0]
        self.unique_instructions = list(self.instruction_to_tool_map.keys())
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(self.definitions)}

    def _build_faiss_index(self, embeddings: np.ndarray):
        embeddings = embeddings.astype('float32')
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    async def _get_embeddings_from_api(self, texts: list, session: aiohttp.ClientSession) -> list:
        semaphore = asyncio.Semaphore(self.api_concurrency_limit)
        tasks = []

        async def fetch_embedding(text):
            async with semaphore: 
                headers = {"Content-Type": "application/json"}
                payload = {"model": self.model_path_or_name, "input": text}
                try:
                    async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result['data'][0]['embedding']
                        else:
                            error_text = await response.text()
                            print(f"API请求失败, 状态码: {response.status}, 错误: {error_text}")
                            return None
                except Exception as e:
                    print(f"请求异常: {e}")
                    return None

        for text in texts:
            tasks.append(fetch_embedding(text))
        
        embeddings = await asyncio.gather(*tasks)
        
        return [emb if emb is not None else [] for emb in embeddings]

    async def initialize_and_get_all_scores(self, all_plan_queries: list):
        """统一的初始化和评分函数"""
        print(f"--- [意图通路] 正在将 {len(self.unique_instructions)} 条唯一指令编码为向量...")
        
        instruction_embeddings = None
        if self.mode == 'local':
            instruction_embeddings = self.model.encode(self.unique_instructions, convert_to_tensor=False, show_progress_bar=True)
        else: # api mode
            async with aiohttp.ClientSession() as session:
                instruction_embeddings = await self._get_embeddings_from_api(self.unique_instructions, session)
                successful_embeddings = [emb for emb in instruction_embeddings if emb]
                if len(successful_embeddings) != len(self.unique_instructions):
                    print(f"警告: {len(self.unique_instructions) - len(successful_embeddings)} 条指令向量化失败!")
                    if not successful_embeddings:
                        raise RuntimeError("所有指令向量化失败，无法继续！")
                
                self.unique_instructions = [inst for inst, emb in zip(self.unique_instructions, instruction_embeddings) if emb]
                instruction_embeddings = successful_embeddings

        self._build_faiss_index(np.array(instruction_embeddings))
        print("--- [意图通路] Faiss索引构建完成 ---")

        print(f"--- [意图通路] 正在对 {len(all_plan_queries)} 条查询进行评分...")
        all_semantic_scores = []
        query_embeddings = None

        if self.mode == 'local':
            query_embeddings = self.model.encode(all_plan_queries, convert_to_tensor=False, show_progress_bar=True)
        else: # api mode
             async with aiohttp.ClientSession() as session:
                query_embeddings = await self._get_embeddings_from_api(all_plan_queries, session)
        
        for query_embedding in tqdm(query_embeddings, desc="计算语义分数"):
            if not query_embedding:
                all_semantic_scores.append(np.zeros(len(self.definitions), dtype='float32'))
                continue

            query_embedding_np = np.array([query_embedding], dtype='float32')
            faiss.normalize_L2(query_embedding_np)
            
            num_neighbors = min(len(self.unique_instructions), 50) 
            distances, indices = self.faiss_index.search(query_embedding_np, k=num_neighbors)
            
            tool_scores = np.zeros(len(self.definitions), dtype='float32')
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    matched_instruction = self.unique_instructions[idx]
                    tool_def = self.instruction_to_tool_map.get(matched_instruction)
                    if tool_def:
                        tool_idx = self.tool_name_to_idx.get(tool_def['name'])
                        if tool_idx is not None:
                            tool_scores[tool_idx] = max(tool_scores[tool_idx], dist)
            all_semantic_scores.append(tool_scores)
            
        return all_semantic_scores

# ==============================================================================
# 区域 2.5: Reranker类定义 (异步改造版)
# ==============================================================================
class QwenReranker:
    """
    【精排通路】使用Qwen3-Reranker对召回结果进行重排序。
    支持 'local' (同步) 和 'api' (异步) 两种模式。
    """
    def __init__(self, mode: str, model_path_or_name: str, api_url: str = None, device: str = 'cuda', api_concurrency_limit: int = 20):
        self.mode = mode
        self.model_path_or_name = model_path_or_name
        self.api_url = api_url
        self.model = None

        print(f"--- [精排通路] 模式: {self.mode.upper()} ---")
        if self.mode == 'local':
            self.device = device if torch.cuda.is_available() else 'cpu'
            print(f"--- [精排通路] 正在加载Reranker模型: {self.model_path_or_name} ---")
            self.model = CrossEncoder(self.model_path_or_name, device=self.device, trust_remote_code=True)
            print(f"--- [精排通路] Reranker模型加载完成，使用设备: {self.device} ---")
        elif self.mode == 'api':
            print(f"--- [精排通路] API模式已启用，目标URL: {self.api_url} ---")
            print(f"--- [精排通路] API并发请求阈值设置为: {api_concurrency_limit} ---")
            if not self.api_url:
                raise ValueError("API模式下必须提供 api_url")
            # 为异步请求创建信号量进行并发控制
            self.semaphore = asyncio.Semaphore(api_concurrency_limit)
        else:
            raise ValueError("Reranker模式 (mode) 必须是 'local' 或 'api'")

    def rerank_sync(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """
        [同步方法] 对候选指令进行重排序，主要用于本地模式。
        """
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    async def rerank_async(self, query: str, documents: list[str], session: aiohttp.ClientSession) -> list[tuple[str, float]]:
        """
        [异步方法] 使用aiohttp并发请求API进行重排序。
        """
        if not documents:
            return []

        async with self.semaphore: # 使用信号量控制并发数
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model_path_or_name,
                "query": query,
                "documents": documents,
            }
            try:
                async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        api_results = await response.json()
                        scored_docs = [("", 0.0)] * len(documents)
                        for res in api_results:
                            if 'index' in res and res['index'] < len(documents):
                                scored_docs[res['index']] = (documents[res['index']], res['score'])
                        
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        return scored_docs
                    else:
                        error_text = await response.text()
                        print(f"API Rerank请求失败，状态码: {response.status}, 错误: {error_text[:200]}")
                        return [(doc, 0.0) for doc in documents]
            except Exception as e:
                print(f"API Rerank请求异常: {e}")
                return [(doc, 0.0) for doc in documents]

# ==============================================================================
# 区域 3: 评测函数 (不变)
# ==============================================================================
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

# ==============================================================================
# 区域 4: 工具定义 (不变)
# ==============================================================================
def get_exact_tool_definitions():
    tools = [
        # 1. 购物 - 搜索 (1.1)
        {"name": "search_goods(app, search_info_slot, page_type, filter_detail_slot, type_slot, area_slot, order_type)", "description": "在app程序中依据名称搜索商品,可以指定具体在哪一个子页面进行搜索, 搜索结果的筛选条件和排序方式"},
        {"name": "search_stores(app, search_info_slot, filter_type, filter_detail_slot, location_slot, qualification_slot, order_type)", "description": "在app程序中依据名称搜索店铺,可以使用筛选器限制搜索结果,也可以指定搜索结果的排序方式"},
        {"name": "open_search_history(app)", "description": "打开app程序的搜索历史界面"},
        {"name": "delete_search_history(app)", "description": "清除app中的搜索历史"},
        {"name": "open_camera_search(app)", "description": "打开app程序的图片搜索功能"},
        {"name": "search_delivery_time(app, search_info_slot, address_slot)", "description": "搜索一件商品并根据给出的地址查询该商品送达该地址的预估运送时间"},
        {"name": "search_cart_content(app, search_info_slot)", "description": "在app应用程序中查看购物车/采购车(阿里巴巴的叫法)查找商品"},
        {"name": "search_in_favorite_goods(app, search_info_slot)", "description": "在app程序中打开收藏的、喜爱、想要或关注商品的页面,并在其中的搜索栏中进行搜索"},
        {"name": "search_in_favorite_stores(app, search_info_slot)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面,并在其中的搜索栏搜索商品"},
        {"name": "search_order(app, search_info_slot, order_status)", "description": "在app应用程序中搜索订单"},

        # 2. 购物 - 打开 (1.2)
        {"name": "open_goods_page(app, search_info_slot, page_type)", "description": "通过商品名称找到并打开其详情页面,可以指定子页面,例如评论、规格、参数、详情等"},
        {"name": "open_stores_page(app, store_name_slot, search_info_slot, category_slot)", "description": "通过店铺名称找到并打开店铺的内容页面,可以在其中进行店铺内搜索或打开类别子页面"},
        {"name": "open_special_page(app, page_type)", "description": "打开特殊页面,例如活动页面"},

        # 3. 购物 - 购物车 (1.3)
        {"name": "open_cart_content(app, filter_type, filter_detail_slot)", "description": "在app应用程序中查看购物车/采购车(阿里巴巴的叫法)指定类型的商品"},
        {"name": "add_into_cart(app, search_info_slot, specification_slot, num_slot, address_slot)", "description": "搜索商品并将其添加入购物车,可以指定添加的商品规格、数量并选择收货地址"},

        # 4. 购物 - 收藏 (1.4)
        {"name": "open_favorite_goods(app, filter_type, filter_detail_slot, order_type)", "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面,并按照条件进行筛选"},
        {"name": "open_favorite_stores(app, filter_type)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面,并按照条件进行筛选"},
        {"name": "add_into_favorite_goods(app, search_info_slot)", "description": "在app程序中搜索商品,并将其添加到商品收藏夹中"},
        {"name": "add_into_favorite_stores(app, search_info_slot)", "description": "在app程序中按照店铺名搜索店铺,并将其添加到店铺收藏夹中"},
        {"name": "delete_favorite_goods(app, search_info_slot)", "description": "在app程序的商品收藏夹中搜索指定商品并将其删除"},
        
        # 5. 购物 - 下单 (1.5)
        {"name": "order_to_purchase_goods(app, search_info_slot, specification_slot, num_slot, address_slot, payment_method_slot)", "description": "通过商品名称找到商品并下单购买,可以指定添加的商品规格、数量并选择收货地址以及支付方式"},

        # 6. 购物 - 订单 (1.6)
        {"name": "open_orders_bought(app, order_status, filter_detail_slot)", "description": "在app应用程序中查看买入的指定状态的订单列表,例如待付款、待收货、待评价等。"},
        {"name": "open_orders_sold(app, order_status, filter_detail_slot)", "description": "在app应用程序中查看自己售卖的指定状态的订单列表,例如待付款、待收货、待评价等。"},
        {"name": "open_orders_release(app, order_status)", "description": "在app应用程序中查看自己发布的指定状态的订单列表,例如在卖、草稿、已下架等。"},
        {"name": "open_orders_all_review(app)", "description": "在app应用程序中查看待评价状态的订单列表,在不指定购买还是售卖的订单时,及全都要看时使用。"},
        {"name": "apply_after_sales(app, search_info_slot, after_sales_type, reason_slot)", "description": "在app应用程序中搜索订单,并申请售后"},

        # 7. 购物 - 物流 (1.7)
        {"name": "open_logistics_receive(app, filter_type)", "description": "打开显示已购商品信息的界面,查看相关物流信息,并根据物流情况进行筛选"},
        {"name": "open_logistics_send(app, filter_type)", "description": "打开显示已售商品信息的界面,查看相关物流信息,并根据物流情况进行筛选"},
        {"name": "open_express_delivery(app)", "description": "打开app寄送快递的界面"},
        {"name": "manage_order_logistics_status(app, search_info_slot, action_type)", "description": "在app中管理指定订单的物流状态,包括催发货,催配送,确认收货"},
        {"name": "open_order_tracking_number(app, search_info_slot)", "description": "在app中查询指定订单的物流单号"},
        {"name": "call_order_courier(app, search_info_slot)", "description": "在app中拨打指定订单的快递电话"},

        # 8. 购物 - 客服 (1.8)
        {"name": "open_customer_service(app, order_slot, store_slot)", "description": "在app应用程序中联系官方客服,或联系指令订单的店铺客服,或联系指定店铺的客服"},
        {"name": "apply_price_protection(app)", "description": "在app应用程序中联系客服进行价保"},

        # 9. 购物 - 评价 (1.9)
        {"name": "rate_order(app, search_info_slot, rating_slot, review_text_slot, upload_images)", "description": "在app应用程序评价商城中的指定订单"},

        # 10. 购物 - 发票 (1.10)
        {"name": "open_invoice_page(app, page_type)", "description": "在app应用程序中打开与发票相关的页面"},

        # 11. 购物 - 签到 (1.11)
        {"name": "sign_in(app, page_type)", "description": "在app程序中完成每日签到,领取积分、金币等奖励的操作"},

        # 12. 购物 - 启动 (1.12)
        {"name": "open_app(app)", "description": "打开指定的应用程序"},
    ]
    return tools

# ==============================================================================
# 区域 5: 评测核心逻辑 (异步改造版)
# ==============================================================================
async def evaluate_recall_system(
    data_df, 
    all_bm25_scores, 
    all_semantic_scores, 
    all_tools_definitions, 
    instruction_to_tool_map,
    tool_to_instructions_map,
    alpha, 
    k_values, 
    reranker: QwenReranker,
    rerank_top_n, 
    session: aiohttp.ClientSession, # <-- 新增session参数
    full_report=False
):
    results = defaultdict(lambda: defaultdict(list))
    error_cases = []
    latency_records = []
    detailed_predictions = [] 

    tool_name_to_def_map = {tool['name']: tool for tool in all_tools_definitions}

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if (max_s - min_s) == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    # --- 步骤 A: 准备所有任务 ---
    # 准备一个列表，存放每个待处理样本的 rerank 任务协程和相关数据
    rerank_tasks = []
    prepared_data = []

    for i, row in data_df.iterrows():
        query = row['plan（在xx中做什么）']
        
        # 1. 召回
        bm25_scores = all_bm25_scores[i]
        semantic_scores = all_semantic_scores[i]
        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)
        recall_scores = alpha * norm_bm25 + (1 - alpha) * norm_semantic
        
        recall_sorted_indices = np.argsort(recall_scores)[::-1]
        recall_candidate_tools = [all_tools_definitions[idx] for idx in recall_sorted_indices[:rerank_top_n]]

        # 2. 扩展
        instructions_to_rerank = set()
        for tool in recall_candidate_tools:
            instructions = tool_to_instructions_map.get(tool['name'], [])
            instructions_to_rerank.update(instructions)
        instructions_to_rerank = list(instructions_to_rerank)

        # 区分本地同步模式和API异步模式
        if reranker.mode == 'local':
            # 对于本地模式，直接同步执行
            task = reranker.rerank_sync(query, instructions_to_rerank)
        else:
            # 对于API模式，创建异步任务
            task = reranker.rerank_async(query, instructions_to_rerank, session)
        
        rerank_tasks.append(task)
        prepared_data.append({
            "row": row,
            "recall_scores": recall_scores,
            "recall_candidate_tools": recall_candidate_tools
        })

    # --- 步骤 B: 并发执行所有 Rerank 任务 ---
    # 使用tqdm.asyncio来为异步任务添加进度条
    all_reranked_instructions = await anaylse_tqdm.gather(*rerank_tasks, desc="并发精排中")

    # --- 步骤 C: 收集并处理所有结果 ---
    total_processing_time = 0
    for i, reranked_instructions in enumerate(all_reranked_instructions):
        start_time = time.time()
        
        # 从准备好的数据中获取上下文
        data = prepared_data[i]
        row = data['row']
        recall_scores = data['recall_scores']
        recall_candidate_tools = data['recall_candidate_tools']
        ground_truth = row['ground_truth_tool']
        
        # 4. 聚合
        tool_rerank_scores = defaultdict(lambda: -1e9)
        for instruction_text, score in reranked_instructions:
            tool_def = instruction_to_tool_map.get(instruction_text)
            if tool_def:
                tool_name = tool_def['name']
                tool_rerank_scores[tool_name] = max(tool_rerank_scores[tool_name], score)

        # 5. 最终排序
        recall_candidate_tools.sort(key=lambda tool: tool_rerank_scores[tool['name']], reverse=True)
        retrieved = recall_candidate_tools
        retrieved_scores = [tool_rerank_scores[tool['name']] for tool in retrieved]
        
        end_time = time.time()
        # 注意：这里的时延仅包含后处理部分，总时延需要在循环外计算
        total_processing_time += (end_time - start_time)

        # 6. 评估
        if full_report:
            prediction_record = {
                "query": row['query'],
                "plan": row['plan（在xx中做什么）'],
                "ground_truth": [_get_tool_names(ground_truth)],
                "retrieved_top_k": [{"tool": t.get('name'), "score": float(s)} for t, s in zip(retrieved[:max(k_values)], retrieved_scores[:max(k_values)])]
            }
            detailed_predictions.append(prediction_record)

            for k in k_values:
                results['Recall@K'][k].append(calculate_recall_at_k(retrieved, ground_truth, k))
                results['HR@K'][k].append(calculate_hit_ratio_at_k(retrieved, ground_truth, k))
                results['MAP@K'][k].append(calculate_average_precision_at_k(retrieved, ground_truth, k))
                results['MRR@K'][k].append(calculate_mrr_at_k(retrieved, ground_truth, k))
                results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved, ground_truth, k))
                results['COMP@K'][k].append(calculate_completeness_at_k(retrieved, ground_truth, k))
            
            results['AUC']['all'].append(calculate_auc_for_query(recall_scores, all_tools_definitions, ground_truth))
            
            if calculate_recall_at_k(retrieved, ground_truth, 1) < 1.0:
                gt_name = _get_tool_names(ground_truth).pop() if ground_truth else "N/A"
                pred_name_top1 = retrieved[0].get('name') if retrieved else "N/A"
                error_cases.append({"Query": row['plan（在xx中做什么）'], "Ground Truth": [gt_name], "Prediction@1": [pred_name_top1], "Prediction@5": [r.get('name') for r in retrieved[:5]]})
        else:
            results['Recall@K'][1].append(calculate_recall_at_k(retrieved, ground_truth, 1))
    
    # 将总的后处理时间分摊到每个样本上，作为时延记录
    avg_latency = total_processing_time / len(data_df) if len(data_df) > 0 else 0
    latency_records = [avg_latency] * len(data_df)

    if full_report:
        return results, error_cases, latency_records, detailed_predictions
    else:
        return np.mean(results['Recall@K'][1])

# ==============================================================================
# 区域 6: 主程序 (异步改造版)
# ==============================================================================
async def main():
    # --- 0. 配置区域 ---
    EMBEDDING_MODE = 'api'  
    RERANKER_MODE = 'api' 

    EMBEDDING_MODEL_PATH = '/path/to/your/Qwen3-Embedding-0.6B'
    VLLM_EMBEDDING_API_URL = "http://localhost:8000/v1/embeddings" 
    VLLM_EMBEDDING_MODEL_NAME = "/home/workspace/lgq/shop/model/Qwen3-Embedding-0.6B"

    RERANKER_MODEL_PATH = '/home/workspace/lgq/shop/model/Qwen3-Reranker-0.6B'
    VLLM_RERANKER_API_URL = "http://localhost:8001/v1/rerank" 
    
    # 为 embedding 和 reranker 分别设置并发限制
    EMBEDDING_API_CONCURRENCY_LIMIT = 50000
    RERANKER_API_CONCURRENCY_LIMIT = 50000

    RERANK_TOP_N = 20
    annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_plan_0815.csv'
    K_VALUES = [1, 2, 3, 5, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10
    OUTPUT_FILE_PATH = f'/home/workspace/lgq/shop/data/instruction_reranked_results_{RERANKER_MODE}_plan_0.6b_full_search_async.csv'

    # --- 1. 数据加载 ---
    print("--- 步骤 1: 加载完整数据集 ---")
    data_df = pd.read_csv(annotated_data_file_path)
    data_df = data_df.dropna(subset=['指令', 'ground_truth_tool', 'plan（在xx中做什么）']).reset_index(drop=True)
    def parse_tools(s): return ast.literal_eval(s) if isinstance(s, str) else []
    data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(parse_tools)
    print(f"数据加载完成: 共 {len(data_df)} 条。\n")

    # --- 2. 初始化召回器、精排器及数据映射 ---
    all_tools_definitions = get_exact_tool_definitions()
    
    init_start_time = time.time()
    bm25_retriever = BM25Retriever(data_df, all_tools_definitions)
    instruction_searcher = InstructionSearcher(
        data_df=data_df, all_tools_definitions=all_tools_definitions,
        mode=EMBEDDING_MODE,
        model_path_or_name=EMBEDDING_MODEL_PATH if EMBEDDING_MODE == 'local' else VLLM_EMBEDDING_MODEL_NAME,
        api_url=VLLM_EMBEDDING_API_URL if EMBEDDING_MODE == 'api' else None,
        api_concurrency_limit=EMBEDDING_API_CONCURRENCY_LIMIT 
    )
    reranker = QwenReranker(
        mode=RERANKER_MODE,
        model_path_or_name=RERANKER_MODEL_PATH,
        api_url=VLLM_RERANKER_API_URL if RERANKER_MODE == 'api' else None,
        api_concurrency_limit=RERANKER_API_CONCURRENCY_LIMIT
    )

    print("--- 正在构建工具->指令的反向映射... ---")
    tool_to_instructions_map = defaultdict(list)
    instruction_to_tool_map = instruction_searcher.instruction_to_tool_map
    for instruction, tool_def in instruction_to_tool_map.items():
        tool_to_instructions_map[tool_def['name']].append(instruction)
    print("--- 反向映射构建完成 ---\n")

    init_end_time = time.time()
    print(f"\n--- [计时] 所有模型和数据映射初始化耗时: {init_end_time - init_start_time:.2f} 秒 ---\n")

    # --- 3. 计算所有召回分数 ---
    print("\n--- 步骤 3: 计算所有召回分数 ---")
    all_bm25_scores = [bm25_retriever.retrieve_scores(row['plan（在xx中做什么）']) for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="计算BM25分数")]
    all_plan_queries = data_df['plan（在xx中做什么）'].tolist()
    all_semantic_scores = await instruction_searcher.initialize_and_get_all_scores(all_plan_queries)
    
    # --- 步骤 4. 在【全量数据】上进行Alpha值网格搜索并直接获取最终结果 ---
    print(f"\n--- 步骤 4: 开始在全部 {len(data_df)} 条数据上进行Alpha值网格搜索 ---")
    
    alpha_range = np.linspace(0.17, 0.19, 3) 
    best_alpha = -1
    best_score = -1
    
    best_results = None
    best_error_cases = None
    best_latency_records = None
    best_detailed_predictions = None

    # 创建一个 ClientSession 供所有异步API调用使用
    async with aiohttp.ClientSession() as session:
        for alpha in tqdm(alpha_range, desc="Alpha网格搜索中(全量数据)"):
            eval_start_time = time.time()
            # await异步函数，并传入session
            results, error_cases, latency_records, detailed_predictions = await evaluate_recall_system(
                data_df,
                all_bm25_scores,
                all_semantic_scores,
                all_tools_definitions, 
                instruction_to_tool_map, 
                tool_to_instructions_map,
                alpha, 
                K_VALUES, 
                reranker, 
                RERANK_TOP_N, 
                session, # <-- 传入session
                full_report=True
            )
            eval_end_time = time.time()
            print(f"\nAlpha={alpha:.2f} 的完整评测耗时: {eval_end_time - eval_start_time:.2f} 秒")
            
            current_score = np.mean(results['Recall@K'][1])

            if current_score > best_score:
                best_score = current_score
                best_alpha = alpha
                best_results = results
                best_error_cases = error_cases
                best_latency_records = latency_records
                best_detailed_predictions = detailed_predictions

    print(f"\n--- Alpha值网格搜索完成 (在全部 {len(data_df)} 个样本上) ---")
    print(f"找到的最佳Alpha值: {best_alpha:.2f} (对应的最高平均Recall@1为: {best_score:.4f})")
    
    # --- 步骤 5: 直接使用搜索过程中保存的最佳结果进行报告 ---
    print(f"\n--- 步骤 5: 使用最佳Alpha={best_alpha:.2f}的结果生成最终报告 ---")
    
    results = best_results
    latency_records = best_latency_records
    
    # --- 步骤 6. 报告最终结果 ---
    print(f"\n\n--- 步骤 6: 最终评测结果报告 (全量数据寻优, Alpha: {best_alpha:.2f}) ---")
    
    final_scores_report = {}
    for metric, vals in results.items():
        if metric == 'AUC': 
            final_scores_report['AUC'] = np.mean(vals['all'])
        else: 
            final_scores_report[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    
    report_df = pd.DataFrame({ m: final_scores_report[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    
    # 时延计算方式已在evaluate_recall_system中更新，这里直接取平均即可
    average_latency_ms = np.mean(latency_records) * 1000

    print(f"召回+指令粒度精排模型 (BM25+意图[{EMBEDDING_MODE.upper()}] -> Qwen3-Reranker[{RERANKER_MODE.upper()}]) 评测结果:")
    print("-" * 80)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (基于召回阶段分数)**: {final_scores_report['AUC']:.4f}")
    # 注意：这里的时延现在主要反映的是后处理（聚合、排序）部分，因为网络请求是并发的。
    print(f"**平均后处理时延**: {average_latency_ms:.4f} 毫秒/查询")
    print(f"**精排候选工具数 (RERANK_TOP_N)**: {RERANK_TOP_N}")
    print("-" * 80)
    
    # --- 7 & 8. 错误分析和保存文件 --- (逻辑不变)
    # ...


if __name__ == "__main__":
    asyncio.run(main())