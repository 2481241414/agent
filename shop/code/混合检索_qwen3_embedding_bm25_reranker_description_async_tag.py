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

# ==============================================================================
# 区域 1: 检查并导入所需库 (已更新)
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.metrics import roc_auc_score
    import aiohttp # 异步HTTP请求
    import asyncio # 异步IO
    from tqdm.asyncio import tqdm as anaylse_tqdm # 异步任务进度条
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install faiss-cpu torch sentence-transformers transformers rank_bm25 scikit-learn pandas tqdm aiohttp")
    exit()

# ==============================================================================
# 区域 2: 召回器类定义 (已迁移为“基于描述”的策略)
# ==============================================================================
class BM25Retriever:
    """【关键词通路】BM25召回器，基于工具描述构建语料库。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, k1=1.5, b=0.75):
        self.definitions = all_tools_definitions
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(all_tools_definitions)}
        self._add_jieba_words()
        
        print("--- [BM25通路] 正在基于【工具描述】构建关键词增强语料库... ---")
        corpus = self._build_corpus_from_descriptions(data_df)
        tokenized_corpus = [jieba.lcut(doc, cut_all=False) for doc in tqdm(corpus, desc="BM25语料库分词")]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        print("--- [BM25通路] 召回器构建完成 ---")

    def _add_jieba_words(self):
        for tool in self.definitions:
            jieba.add_word(tool.get('name', '').split('(')[0], freq=100)
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "发票", "优惠券"]
        for word in core_words:
            jieba.add_word(word, freq=100)

    def _build_corpus_from_descriptions(self, data_df: pd.DataFrame) -> list:
        """
        迁移自新参考代码：为每个工具构建一个包含其官方描述和所有相关标注描述的文档。
        """
        tool_text_aggregator = defaultdict(list)
        for _, row in data_df.iterrows():
            tools = row.get('ground_truth_tool')
            descriptions_str = row.get('ground_truth_tool_description')

            if not isinstance(tools, list) or not tools or pd.isna(descriptions_str):
                continue
            
            descriptions = descriptions_str.strip().split('\n')
            
            if len(tools) == len(descriptions):
                for tool, desc in zip(tools, descriptions):
                    tool_name = tool['name']
                    tool_text_aggregator[tool_name].append(desc)
                    tool_text_aggregator[tool_name].append(tool_name.split('(')[0])
            else:
                for tool in tools:
                    tool_name = tool['name']
                    tool_text_aggregator[tool_name].extend(descriptions)
                    tool_text_aggregator[tool_name].append(tool_name.split('(')[0])
        
        corpus = [''] * len(self.definitions)
        for tool_def in self.definitions:
            tool_name = tool_def['name']
            tool_idx = self.tool_name_to_idx[tool_name]
            aggregated_text = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
            print(aggregated_text)
            document = f"{aggregated_text}"
            corpus[tool_idx] = document
            
        return corpus

    def retrieve_scores(self, query: str) -> np.ndarray:
        tokenized_query = jieba.lcut(query, cut_all=False)
        return self.bm25.get_scores(tokenized_query)


class DescriptionSearcher:
    """【精准意图通路】基于工具描述进行向量化和召回。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, mode: str, model_path_or_name: str, api_url: str = None, api_concurrency_limit: int = 10):
        self.definitions = all_tools_definitions
        self.mode = mode
        self.model_path_or_name = model_path_or_name
        self.api_url = api_url
        self.model = None
        self.faiss_index = None
        self.api_concurrency_limit = api_concurrency_limit

        print(f"--- [意图通路] 模式: {self.mode.upper()} ---")
        if self.mode == 'local':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"--- [意图通路] 正在使用设备: {self.device} ---")
            self.model = SentenceTransformer(self.model_path_or_name, trust_remote_code=True, device=self.device)
        elif self.mode == 'api':
            print(f"--- [意图通路] API并发请求阈值设置为: {self.api_concurrency_limit} ---")
        else:
            raise ValueError("模式 (mode) 必须是 'local' 或 'api'")
        
        self._build_mappings_from_description(data_df)

    def _build_mappings_from_description(self, data_df: pd.DataFrame):
        """
        迁移自新参考代码：构建从“工具描述”到工具定义的映射。
        """
        self.description_to_tool_map = {}
        for _, row in data_df.iterrows():
            tools = row.get('ground_truth_tool')
            descriptions_str = row.get('ground_truth_tool_description')
            
            if pd.isna(descriptions_str) or not isinstance(tools, list) or not tools:
                continue
            
            descriptions = [d.strip() for d in descriptions_str.strip().split('\n')]
            
            if len(descriptions) == len(tools):
                for desc, tool in zip(descriptions, tools):
                    self.description_to_tool_map[desc] = tool
            elif descriptions and tools: # 降级策略
                self.description_to_tool_map[descriptions[0]] = tools[0]
        
        self.unique_descriptions = list(self.description_to_tool_map.keys())
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(self.definitions)}

    def _build_faiss_index(self, embeddings: np.ndarray):
        embeddings = embeddings.astype('float32')
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    async def _get_embeddings_from_api(self, texts: list, session: aiohttp.ClientSession) -> list:
        semaphore = asyncio.Semaphore(self.api_concurrency_limit)
        
        async def fetch_embedding(text):
            if not text or text.isspace(): return None
            async with semaphore:
                headers = {"Content-Type": "application/json"}
                payload = {"model": self.model_path_or_name, "input": text}
                try:
                    async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get('data', [{}])[0].get('embedding')
                        else:
                            error_text = await response.text()
                            print(f"API请求失败, 状态码: {response.status}, 错误: {error_text[:200]}, 失败文本: '{text}'")
                            return None
                except Exception as e:
                    print(f"API请求时发生未知异常: {e}, 失败文本: '{text}'")
                    return None

        tasks = [fetch_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def initialize_and_get_all_scores(self, all_queries: list):
        print(f"--- [意图通路] 正在将 {len(self.unique_descriptions)} 条唯一【工具描述】编码为向量...")
        
        description_embeddings = []
        if self.mode == 'local':
            description_embeddings = self.model.encode(self.unique_descriptions, convert_to_tensor=False, show_progress_bar=True)
        else:
            async with aiohttp.ClientSession() as session:
                raw_embeddings = await self._get_embeddings_from_api(self.unique_descriptions, session)
            
            successful_embeddings_map = {}
            for desc, emb in zip(self.unique_descriptions, raw_embeddings):
                if emb: successful_embeddings_map[desc] = emb

            if len(successful_embeddings_map) != len(self.unique_descriptions):
                 print(f"警告: {len(self.unique_descriptions) - len(successful_embeddings_map)} 条【工具描述】向量化失败!")
                 if not successful_embeddings_map: raise RuntimeError("所有【工具描述】向量化失败，无法继续！")
            
            self.unique_descriptions = list(successful_embeddings_map.keys())
            description_embeddings = list(successful_embeddings_map.values())
            self.description_to_tool_map = {k: self.description_to_tool_map[k] for k in self.unique_descriptions}

        self._build_faiss_index(np.array(description_embeddings))
        print("--- [意图通路] Faiss索引构建完成 ---")

        print(f"--- [意图通路] 正在对 {len(all_queries)} 条查询进行评分...")
        query_embeddings = None
        if self.mode == 'local':
            query_embeddings = self.model.encode(all_queries, convert_to_tensor=False, show_progress_bar=True)
        else:
             async with aiohttp.ClientSession() as session:
                query_embeddings = await self._get_embeddings_from_api(all_queries, session)
        
        all_semantic_scores = []
        for query_embedding in tqdm(query_embeddings, desc="计算语义分数"):
            if not query_embedding:
                all_semantic_scores.append(np.zeros(len(self.definitions), dtype='float32'))
                continue

            query_embedding_np = np.array([query_embedding], dtype='float32')
            faiss.normalize_L2(query_embedding_np)
            
            num_neighbors = min(len(self.unique_descriptions), 50) 
            distances, indices = self.faiss_index.search(query_embedding_np, k=num_neighbors)
            
            tool_scores = np.zeros(len(self.definitions), dtype='float32')
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    matched_description = self.unique_descriptions[idx]
                    tool_def = self.description_to_tool_map.get(matched_description)
                    if tool_def and tool_def.get('name') in self.tool_name_to_idx:
                        tool_idx = self.tool_name_to_idx[tool_def['name']]
                        tool_scores[tool_idx] = max(tool_scores[tool_idx], dist)
            all_semantic_scores.append(tool_scores)
            
        return all_semantic_scores

# ==============================================================================
# 区域 2.5: Reranker类定义 (保持不变)
# ==============================================================================
class QwenReranker:
    """【精排通路】使用异步API请求与vLLM rerank端点交互。"""
    def __init__(self, mode: str, model_path_or_name: str, api_url: str = None, api_concurrency_limit: int = 20):
        self.mode = mode
        self.model_path_or_name = model_path_or_name
        self.api_url = api_url

        print(f"--- [精排通路] 模式: {self.mode.upper()} ---")
        if self.mode == 'local':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"--- [精排通路] 正在加载本地Reranker模型: {self.model_path_or_name} ---")
            self.model = CrossEncoder(self.model_path_or_name, device=self.device, trust_remote_code=True)
            print(f"--- [精排通路] Reranker模型加载完成，使用设备: {self.device} ---")
        elif self.mode == 'api':
            print(f"--- [精排通路] API模式已启用，目标URL: {self.api_url} ---")
            if not self.api_url: raise ValueError("API模式下必须提供 api_url")
            self.semaphore = asyncio.Semaphore(api_concurrency_limit)
        else:
            raise ValueError("Reranker模式 (mode) 必须是 'local' 或 'api'")

    def rerank_sync(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        if not query or not documents: return []
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        return sorted(list(zip(documents, scores)), key=lambda x: x[1], reverse=True)

    async def rerank_async(self, query: str, documents: list[str], session: aiohttp.ClientSession) -> list[tuple[str, float]]:
        if not query or not documents: return []
        async with self.semaphore:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model_path_or_name,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "return_documents": False
            }
            try:
                async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        api_results = response_json.get('data', [])
                        return sorted(
                            [(documents[res['index']], res['relevance_score']) for res in api_results],
                            key=lambda x: x[1], 
                            reverse=True
                        )
                    else:
                        error_text = await response.text()
                        print(f"API Rerank请求失败，状态码: {response.status}, 错误: {error_text}")
                        return [(doc, 0.0) for doc in documents]
            except Exception as e:
                print(f"客户端请求Rerank API时发生异常: {e}")
                return [(doc, 0.0) for doc in documents]
        
# ==============================================================================
# 区域 3: 评测函数 (保持不变)
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
# 区域 4: 工具定义 (保持不变)
# ==============================================================================
def get_exact_tool_definitions():
    # ... (工具定义列表内容与原代码相同，为节省空间已折叠)
    tools = [
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
        {"name": "open_goods_page(app, search_info_slot, page_type)", "description": "通过商品名称找到并打开其详情页面,可以指定子页面,例如评论、规格、参数、详情等"},
        {"name": "open_stores_page(app, store_name_slot, search_info_slot, category_slot)", "description": "通过店铺名称找到并打开店铺的内容页面,可以在其中进行店铺内搜索或打开类别子页面"},
        {"name": "open_special_page(app, page_type)", "description": "打开特殊页面,例如活动页面"},
        {"name": "open_cart_content(app, filter_type, filter_detail_slot)", "description": "在app应用程序中查看购物车/采购车(阿里巴巴的叫法)指定类型的商品"},
        {"name": "add_into_cart(app, search_info_slot, specification_slot, num_slot, address_slot)", "description": "搜索商品并将其添加入购物车,可以指定添加的商品规格、数量并选择收货地址"},
        {"name": "open_favorite_goods(app, filter_type, filter_detail_slot, order_type)", "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面,并按照条件进行筛选"},
        {"name": "open_favorite_stores(app, filter_type)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面,并按照条件进行筛选"},
        {"name": "add_into_favorite_goods(app, search_info_slot)", "description": "在app程序中搜索商品,并将其添加到商品收藏夹中"},
        {"name": "add_into_favorite_stores(app, search_info_slot)", "description": "在app程序中按照店铺名搜索店铺,并将其添加到店铺收藏夹中"},
        {"name": "delete_favorite_goods(app, search_info_slot)", "description": "在app程序的商品收藏夹中搜索指定商品并将其删除"},
        {"name": "order_to_purchase_goods(app, search_info_slot, specification_slot, num_slot, address_slot, payment_method_slot)", "description": "通过商品名称找到商品并下单购买,可以指定添加的商品规格、数量并选择收货地址以及支付方式"},
        {"name": "open_orders_bought(app, order_status, filter_detail_slot)", "description": "在app应用程序中查看买入的指定状态的订单列表,例如待付款、待收货、待评价等。"},
        {"name": "open_orders_sold(app, order_status, filter_detail_slot)", "description": "在app应用程序中查看自己售卖的指定状态的订单列表,例如待付款、待收货、待评价等。"},
        {"name": "open_orders_release(app, order_status)", "description": "在app应用程序中查看自己发布的指定状态的订单列表,例如在卖、草稿、已下架等。"},
        {"name": "open_orders_all_review(app)", "description": "在app应用程序中查看待评价状态的订单列表,在不指定购买还是售卖的订单时,及全都要看时使用。"},
        {"name": "apply_after_sales(app, search_info_slot, after_sales_type, reason_slot)", "description": "在app应用程序中搜索订单,并申请售后"},
        {"name": "open_logistics_receive(app, filter_type)", "description": "打开显示已购商品信息的界面,查看相关物流信息,并根据物流情况进行筛选"},
        {"name": "open_logistics_send(app, filter_type)", "description": "打开显示已售商品信息的界面,查看相关物流信息,并根据物流情况进行筛选"},
        {"name": "open_express_delivery(app)", "description": "打开app寄送快递的界面"},
        {"name": "manage_order_logistics_status(app, search_info_slot, action_type)", "description": "在app中管理指定订单的物流状态,包括催发货,催配送,确认收货"},
        {"name": "open_order_tracking_number(app, search_info_slot)", "description": "在app中查询指定订单的物流单号"},
        {"name": "call_order_courier(app, search_info_slot)", "description": "在app中拨打指定订单的快递电话"},
        {"name": "open_customer_service(app, order_slot, store_slot)", "description": "在app应用程序中联系官方客服,或联系指令订单的店铺客服,或联系指定店铺的客服"},
        {"name": "apply_price_protection(app)", "description": "在app应用程序中联系客服进行价保"},
        {"name": "rate_order(app, search_info_slot, rating_slot, review_text_slot, upload_images)", "description": "在app应用程序评价商城中的指定订单"},
        {"name": "open_invoice_page(app, page_type)", "description": "在app应用程序中打开与发票相关的页面"},
        {"name": "sign_in(app, page_type)", "description": "在app程序中完成每日签到,领取积分、金币等奖励的操作"},
        {"name": "open_app(app)", "description": "打开指定的应用程序"},
    ]
    return tools

# ==============================================================================
# 区域 5: 评测核心逻辑 (已重构以支持“基于描述”的精排)
# ==============================================================================
async def evaluate_system_with_reranker(
    subset_df, 
    all_bm25_scores, 
    all_semantic_scores, 
    all_tools_definitions,
    reranker: QwenReranker,
    rerank_top_n: int,
    description_searcher: DescriptionSearcher, # 传入DescriptionSearcher实例
    session: aiohttp.ClientSession,
    alpha: float, 
    k_values: list, 
    full_report: bool = False
):
    results = defaultdict(lambda: defaultdict(list))
    error_cases, detailed_predictions, latency_records = [], [], []

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        return np.zeros_like(scores) if (max_s - min_s) == 0 else (scores - min_s) / (max_s - min_s)
    
    # 构建“工具名 -> 官方描述”的映射，用于精排
    tool_to_official_desc_map = {tool['name']: tool['description'] for tool in all_tools_definitions}
    
    rerank_pipeline_start_time = time.time()
    
    original_indices = subset_df.index.tolist()
    rerank_tasks, prepared_data = [], []

    # 1. 准备所有查询的召回候选和精排任务
    for i, (_, row) in enumerate(subset_df.iterrows()):
        original_index = original_indices[i]
        query = row['query'] # 使用原始query进行精排，相关性最直接
        
        bm25_scores = all_bm25_scores[original_index]; semantic_scores = all_semantic_scores[original_index]
        recall_scores = alpha * normalize(bm25_scores) + (1 - alpha) * normalize(semantic_scores)
        
        recall_sorted_indices = np.argsort(recall_scores)[::-1]
        recall_candidate_tools = [all_tools_definitions[idx] for idx in recall_sorted_indices[:rerank_top_n]]

        # 准备精排文档：使用候选工具的“官方描述”
        docs_to_rerank = [tool_to_official_desc_map.get(t['name']) for t in recall_candidate_tools if tool_to_official_desc_map.get(t['name'])]
        
        task = reranker.rerank_sync(query, docs_to_rerank) if reranker.mode == 'local' else reranker.rerank_async(query, docs_to_rerank, session)
        
        rerank_tasks.append(task)
        prepared_data.append({"row": row, "recall_scores": recall_scores, "recall_candidate_tools": recall_candidate_tools})

    # 2. 并发执行所有精排任务
    desc = f"并发精排中 (Mode: {reranker.mode.upper()})"
    all_reranked_results = await anaylse_tqdm.gather(*rerank_tasks, desc=desc) if reranker.mode == 'api' else rerank_tasks

    if len(subset_df) > 0:
        avg_latency = (time.time() - rerank_pipeline_start_time) / len(subset_df)
        latency_records = [avg_latency] * len(subset_df)

    # 3. 处理精排结果并计算指标
    # 构建“官方描述 -> 工具名”的反向映射
    official_desc_to_tool_name_map = {v: k for k, v in tool_to_official_desc_map.items()}
    for i, reranked_descriptions in enumerate(all_reranked_results):
        data = prepared_data[i]
        row, recall_scores, recall_candidate_tools = data['row'], data['recall_scores'], data['recall_candidate_tools']
        
        tool_rerank_scores = defaultdict(lambda: -1e9)
        for desc_text, score in reranked_descriptions:
            tool_name = official_desc_to_tool_name_map.get(desc_text)
            if tool_name:
                tool_rerank_scores[tool_name] = max(tool_rerank_scores[tool_name], score)

        retrieved = sorted(recall_candidate_tools, key=lambda tool: tool_rerank_scores.get(tool['name'], -1e9), reverse=True)
        retrieved_scores = [tool_rerank_scores.get(tool.get('name'), -1e9) for tool in retrieved]

        if full_report:
            ground_truth = row['ground_truth_tool']
            prediction_record = {"query": row['query'], "plan": row['plan（在xx中做什么）'], "ground_truth": list(_get_tool_names(ground_truth)), "retrieved_top_k": [{"tool": t.get('name'), "score": float(s)} for t, s in zip(retrieved[:max(k_values)], retrieved_scores[:max(k_values)])]}
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
                error_cases.append({"Query": row['query'], "Ground Truth": list(_get_tool_names(ground_truth)), "Prediction@1": [retrieved[0].get('name') if retrieved else "N/A"], "Prediction@5": [r.get('name') for r in retrieved[:5]]})
        else:
            results['Recall@K'][1].append(calculate_recall_at_k(retrieved, row['ground_truth_tool'], 1))

    return (results, error_cases, latency_records, detailed_predictions) if full_report else np.mean(results['Recall@K'][1])

# ==============================================================================
# 区域 6: 主程序 (已更新以适应新流程)
# ==============================================================================
async def run_full_evaluation_on_subset(
    subset_df, subset_name, all_bm25_scores, all_semantic_scores, all_tools_definitions,
    reranker, description_searcher, session, k_values, rerank_top_n, 
    num_error_examples, output_file_path=None
):
    if subset_df.empty:
        print(f"\n--- 数据子集 '{subset_name}' 为空，跳过评测。 ---")
        return

    print(f"\n\n{'='*30}\n--- 开始对【{subset_name}】子集 (共 {len(subset_df)} 条) 进行评测 ---\n{'='*30}")

    print(f"\n--- 步骤 4 ({subset_name}): 开始进行Alpha值网格搜索 (仅评估Recall@1) ---")
    alpha_range, best_alpha, best_score = np.linspace(0, 1, 11), -1, -1
    
    for alpha in tqdm(alpha_range, desc=f"Alpha网格搜索 ({subset_name})"):
        current_score = await evaluate_system_with_reranker(
            subset_df, all_bm25_scores, all_semantic_scores, all_tools_definitions,
            reranker, rerank_top_n, description_searcher, session,
            alpha, k_values, full_report=False
        )
        if current_score > best_score: best_score, best_alpha = current_score, alpha

    print(f"\n--- Alpha值网格搜索完成 ({subset_name}) ---")
    print(f"找到的最佳Alpha值: {best_alpha:.2f} (对应的最高平均Recall@1为: {best_score:.4f})")
    
    print(f"\n--- 步骤 5 ({subset_name}): 使用最佳Alpha={best_alpha:.2f}进行最终的完整评测 ---")
    results, error_cases, latency_records, detailed_predictions = await evaluate_system_with_reranker(
        subset_df, all_bm25_scores, all_semantic_scores, all_tools_definitions,
        reranker, rerank_top_n, description_searcher, session,
        best_alpha, k_values, full_report=True
    )
    
    print(f"\n\n--- 步骤 6: 最终评测结果报告 (子集: {subset_name}, Alpha: {best_alpha:.2f}) ---")
    final_scores_report = {}
    for metric, vals in results.items():
        if metric == 'AUC': final_scores_report['AUC'] = np.mean(vals['all'])
        else: final_scores_report[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    report_df = pd.DataFrame({m: final_scores_report.get(m, {}) for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T.reindex(columns=[f"@{k}" for k in k_values])
    average_latency_ms = np.mean(latency_records) * 1000
    print(f"召回+精排模型 (Reranker Top-N: {rerank_top_n}) 在【{subset_name}】数据集上的评测结果:")
    print("-" * 70)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (基于召回阶段分数)**: {final_scores_report.get('AUC', 0.0):.4f}")
    print(f"**平均端到端时延 (精排阶段)**: {average_latency_ms:.4f} 毫秒/查询")
    print("-" * 70)

    print(f"\n\n--- 步骤 7 ({subset_name}): Top-1 错误案例分析 (共 {len(error_cases)} 个错误) ---")
    if not error_cases: print(f"🎉 恭喜！在【{subset_name}】数据集上没有发现 Top-1 错误案例！")
    else:
        for i, case in enumerate(error_cases[:num_error_examples]):
            print(f"\n--- 错误案例 {i+1}/{len(error_cases)} ---\n  [查询 Query]: {case['Query']}\n  [真实工具 Ground Truth]: {case['Ground Truth']}\n  [预测工具 Prediction@1]: {case['Prediction@1']}\n  [预测工具 Prediction@5]: {case['Prediction@5']}")
        if len(error_cases) > num_error_examples: print(f"\n... (仅显示前 {num_error_examples} 个错误案例) ...")
    print("-" * 70)

    if output_file_path:
        print(f"\n\n--- 步骤 8 ({subset_name}): 保存详细召回结果到文件 ---")
        try:
            output_records = [{'query': p['query'], 'plan': p['plan'], 'ground_truth': ', '.join(p['ground_truth']), **{f'pred_tool_{i+1}': ti['tool'] for i, ti in enumerate(p['retrieved_top_k'])}, **{f'pred_score_{i+1}': ti['score'] for i, ti in enumerate(p['retrieved_top_k'])}} for p in detailed_predictions]
            pd.DataFrame(output_records).to_csv(output_file_path, index=False, encoding='utf-8-sig')
            print(f"✅ [{subset_name}] 召回结果已成功保存到: {output_file_path}")
        except Exception as e:
            print(f"❌ [{subset_name}] 保存召回结果失败: {e}")
        print("-" * 70)


async def main():
    # --- 0. 配置区域 ---
    EMBEDDING_MODE = 'api'
    EMBEDDING_MODEL_PATH = '/path/to/your/local/Qwen-Embedding-model' 
    VLLM_EMBEDDING_API_URL = "http://localhost:8000/v1/embeddings"
    VLLM_SERVED_EMBEDDING_MODEL_NAME = "/home/workspace/lgq/shop/model/Qwen3-Embedding-8B" # vLLM中部署的Embedding模型名
    EMBEDDING_API_CONCURRENCY_LIMIT = 20

    RERANKER_MODE = 'api' 
    RERANKER_MODEL_PATH = '/home/workspace/lgq/shop/model/Qwen3-Reranker-8B'
    VLLM_RERANK_API_URL = "http://localhost:8001/v1/rerank" 
    RERANKER_API_CONCURRENCY_LIMIT = 20
    RERANK_TOP_N = 5

    # 数据与评测配置
    annotated_data_file_path = '/home/workspace/lgq/shop/data/tagged_cleaned_ground_truth_with_desc_output.csv'
    K_VALUES = [1, 2, 3, 5, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 5
    base_output_path = f'./desc_based_reranked_results_{RERANKER_MODE}'

    # --- 1. 数据加载与划分 ---
    print("--- 步骤 1: 加载并划分数据集 ---")
    try:
        data_df = pd.read_csv(annotated_data_file_path, usecols=['tag', 'query', 'plan（在xx中做什么）', 'ground_truth_tool', 'ground_truth_tool_description'])
    except Exception as e:
        print(f"加载CSV时出错: {e}。请确保文件包含所需列。")
        return
    data_df = data_df.dropna(subset=['ground_truth_tool', 'ground_truth_tool_description', 'tag']).reset_index(drop=True)
    data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
    single_task_df = data_df[data_df['tag'] == '单任务'].copy(); multi_task_df = data_df[data_df['tag'] == '多任务'].copy()
    print(f"数据加载完成: 共 {len(data_df)} 条 (单任务: {len(single_task_df)}, 多任务: {len(multi_task_df)})。\n")

    # --- 2. 初始化召回器、精排器 ---
    all_tools_definitions = get_exact_tool_definitions()
    
    init_start_time = time.time()
    bm25_retriever = BM25Retriever(data_df, all_tools_definitions)
    description_searcher = DescriptionSearcher(
        data_df=data_df, all_tools_definitions=all_tools_definitions, mode=EMBEDDING_MODE,
        model_path_or_name=EMBEDDING_MODEL_PATH if EMBEDDING_MODE == 'local' else VLLM_SERVED_EMBEDDING_MODEL_NAME,
        api_url=VLLM_EMBEDDING_API_URL if EMBEDDING_MODE == 'api' else None,
        api_concurrency_limit=EMBEDDING_API_CONCURRENCY_LIMIT 
    )
    reranker = QwenReranker(
        mode=RERANKER_MODE, model_path_or_name=RERANKER_MODEL_PATH, 
        api_url=VLLM_RERANK_API_URL, api_concurrency_limit=RERANKER_API_CONCURRENCY_LIMIT
    )
    init_end_time = time.time()
    print(f"\n--- [计时] 所有模型和数据映射初始化耗时: {init_end_time - init_start_time:.2f} 秒 ---\n")

    # --- 3. 为【全量数据】一次性计算召回分数 ---
    print("\n--- 步骤 3: 为全量数据计算所有召回分数 (BM25 + 语义) ---")
    # 使用'plan'作为召回查询，因为它更能代表意图
    recall_queries = data_df['plan（在xx中做什么）'].fillna('').tolist()
    all_bm25_scores = [bm25_retriever.retrieve_scores(q) for q in tqdm(recall_queries, desc="计算BM25分数")]
    all_semantic_scores = await description_searcher.initialize_and_get_all_scores(recall_queries)
    
    # --- 4. 对每个子集分别运行包含精排的完整评测流程 ---
    async with aiohttp.ClientSession() as session:
        subsets_to_evaluate = {"单任务": single_task_df, "多任务": multi_task_df, "整体": data_df}
        for name, df in subsets_to_evaluate.items():
            await run_full_evaluation_on_subset(
                subset_df=df, subset_name=name, all_bm25_scores=all_bm25_scores,
                all_semantic_scores=all_semantic_scores, all_tools_definitions=all_tools_definitions,
                reranker=reranker, description_searcher=description_searcher, session=session,
                k_values=K_VALUES, rerank_top_n=RERANK_TOP_N, num_error_examples=NUM_ERROR_EXAMPLES_TO_PRINT,
                output_file_path=f"{base_output_path}_{name.lower()}.csv"
            )

if __name__ == "__main__":
    asyncio.run(main())
