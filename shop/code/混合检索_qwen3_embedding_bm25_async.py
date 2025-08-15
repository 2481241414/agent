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
# 区域 1: 检查并导入所需库 (新增 aiohttp 和 asyncio)
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer 
    import torch
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.metrics import roc_auc_score
    import aiohttp # 异步HTTP请求
    import asyncio # 异步IO
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install faiss-cpu torch sentence-transformers transformers rank_bm25 scikit-learn pandas tqdm aiohttp")
    exit()

# ==============================================================================
# 区域 2: 召回器类定义 (BM25Retriever 不变, InstructionSearcher 重构)
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
            # print(corpus)
        return corpus

    def retrieve_scores(self, query: str) -> np.ndarray:
        tokenized_query = jieba.lcut(query, cut_all=False)
        # print(tokenized_query)
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

    # def _build_faiss_index(self, embeddings: np.ndarray):
    #     embeddings = embeddings.astype('float32')
    #     embedding_dim = embeddings.shape[1]
        
    #     # --- 原有代码 ---
    #     # self.faiss_index = faiss.IndexFlatIP(embedding_dim)
    #     # faiss.normalize_L2(embeddings)
    #     # self.faiss_index.add(embeddings)

    #     # --- 替换为 IndexIVFFlat ---
    #     nlist = 1024  # 划分的区域数量，经验值是向量总数的平方根附近
    #     quantizer = faiss.IndexFlatIP(embedding_dim) # 用一个精确索引来定义区域中心
        
    #     # 使用内积作为度量
    #     self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)

    #     faiss.normalize_L2(embeddings) # 归一化依然重要

    #     print("--- [意图通路] 正在训练IVF索引... ---")
    #     self.faiss_index.train(embeddings)
    #     print("--- [意图通路] 训练完成，正在添加向量... ---")
    #     self.faiss_index.add(embeddings)
        
    #     # 在搜索时，需要设置 nprobe 参数
    #     self.faiss_index.nprobe = 10 # 每次搜索时要查找的区域数量，越大越准但越慢

    def _build_faiss_index(self, embeddings: np.ndarray):
        embeddings = embeddings.astype('float32')
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    async def _get_embeddings_from_api(self, texts: list, session: aiohttp.ClientSession) -> list:
        # 创建 Semaphore 实例，控制并发数量
        semaphore = asyncio.Semaphore(self.api_concurrency_limit)
        tasks = []

        # 定义一个内部函数来包含单次请求的逻辑，并用 semaphore 包裹
        async def fetch_embedding(text):
            async with semaphore: # 在进入前会获取一个"通行证"
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
                            return None # 返回 None 表示失败
                except Exception as e:
                    print(f"请求异常: {e}")
                    return None # 返回 None 表示失败

        # 为每个文本创建一个受控的异步任务
        for text in texts:
            tasks.append(fetch_embedding(text))
        
        # 并发执行所有任务，但实际上同时运行的数量受 semaphore 控制
        embeddings = await asyncio.gather(*tasks)
        
        # 将失败的请求（None）替换为空列表，以保持与后续代码的兼容性
        return [emb if emb is not None else [] for emb in embeddings]

    async def initialize_and_get_all_scores(self, all_plan_queries: list):
        """统一的初始化和评分函数"""
        print(f"--- [意图通路] 正在将 {len(self.unique_instructions)} 条唯一指令编码为向量...")
        
        # --- 步骤1: 向量化指令库 ---
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
                
                # 使用成功的向量构建索引
                self.unique_instructions = [inst for inst, emb in zip(self.unique_instructions, instruction_embeddings) if emb]
                instruction_embeddings = successful_embeddings

        self._build_faiss_index(np.array(instruction_embeddings))
        print("--- [意图通路] Faiss索引构建完成 ---")

        # --- 步骤2: 向量化所有查询并计算分数 ---
        print(f"--- [意图通路] 正在对 {len(all_plan_queries)} 条查询进行评分...")
        all_semantic_scores = []
        query_embeddings = None

        if self.mode == 'local':
            query_embeddings = self.model.encode(all_plan_queries, convert_to_tensor=False, show_progress_bar=True)
        else: # api mode
             async with aiohttp.ClientSession() as session:
                query_embeddings = await self._get_embeddings_from_api(all_plan_queries, session)
        
        for query_embedding in tqdm(query_embeddings, desc="计算语义分数"):
            if not query_embedding: # 如果某个查询向量化失败（返回的是空列表）
                all_semantic_scores.append(np.zeros(len(self.definitions), dtype='float32'))
                continue

            query_embedding_np = np.array([query_embedding], dtype='float32')
            faiss.normalize_L2(query_embedding_np)
            
            num_neighbors = min(len(self.unique_instructions), 30) 
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
# 区域 3: 评测函数
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
# 区域 4: 工具定义
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
# 区域 5: 评测核心逻辑 
# ==============================================================================
def evaluate_recall_system(data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, k_values, full_report=False):
    results = defaultdict(lambda: defaultdict(list))
    error_cases = []
    latency_records = []
    detailed_predictions = [] 

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if (max_s - min_s) == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    for i, (_, row) in enumerate(data_df.iterrows()):
        start_time = time.time()

        ground_truth = row['ground_truth_tool']
        bm25_scores = all_bm25_scores[i]
        semantic_scores = all_semantic_scores[i]
        # print(bm25_scores)
        # print(semantic_scores)
        
        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)
        final_scores = alpha * norm_bm25 + (1 - alpha) * norm_semantic
        
        sorted_indices = np.argsort(final_scores)[::-1]
        retrieved = [all_tools_definitions[i] for i in sorted_indices]
        retrieved_scores = final_scores[sorted_indices]
        
        end_time = time.time()
        latency_records.append(end_time - start_time)

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
            results['AUC']['all'].append(calculate_auc_for_query(final_scores, all_tools_definitions, ground_truth))
            
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
# 区域 6: 主程序 
# ==============================================================================
async def main():
    # --- 0. 配置区域 ---
    # 在这里选择模式: 'local' 或 'api'
    MODE = 'api'  

    # 模型和API的配置
    MODEL_PATH = '/home/workspace/lgq/shop/model/Qwen3-Embedding-0.6B'
    VLLM_API_URL = "http://localhost:8000/v1/embeddings" 
    VLLM_SERVED_MODEL_NAME = "/home/workspace/lgq/shop/model/Qwen3-Embedding-8B" # 这个名字需要和vLLM启动时感知的模型名一致
    # API并发请求阈值，可根据服务器能力调整
    API_CONCURRENCY_LIMIT = 200 # 任何时候最多只有 200000 个请求在同时进行

    # 其他配置
    # annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_plan_0815.csv'
    # annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_fc_0815_能力.csv'
    annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_购物语料-测试结果标注 - 场景单任务1_0815_processed_data.csv'
    K_VALUES = [1, 2, 3, 5, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10
    # OUTPUT_FILE_PATH = f'/home/workspace/lgq/shop/data/evaluate/hybrid_recall_results_{MODE}_plan_0.6b.csv' 
    # OUTPUT_FILE_PATH = f'/home/workspace/lgq/shop/data/evaluate/hybrid_recall_results_{MODE}_fc能力_0.6b.csv'
    OUTPUT_FILE_PATH = f'/home/workspace/lgq/shop/data/evaluate/hybrid_recall_results_{MODE}_测试_8b.csv'

    # --- 1. 数据加载 ---
    print("--- 步骤 1: 加载完整数据集 ---")
    data_df = pd.read_csv(annotated_data_file_path)
    data_df = data_df.dropna(subset=['指令', 'ground_truth_tool']).reset_index(drop=True)
    def parse_tools(s): return ast.literal_eval(s) if isinstance(s, str) else []
    data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(parse_tools)
    print(f"数据加载完成: 共 {len(data_df)} 条。\n")

    # --- 2. 初始化双路召回器 ---
    all_tools_definitions = get_exact_tool_definitions()
    
    init_start_time = time.time()
    bm25_retriever = BM25Retriever(data_df, all_tools_definitions)
    # print(f"data{data_df}")
    # print(f"tool{all_tools_definitions}")
    # print(f"bm25{bm25_retriever}")
    
    instruction_searcher = InstructionSearcher(
        data_df=data_df,
        all_tools_definitions=all_tools_definitions,
        mode=MODE,
        model_path_or_name=MODEL_PATH if MODE == 'local' else VLLM_SERVED_MODEL_NAME,
        api_url=VLLM_API_URL if MODE == 'api' else None,
        api_concurrency_limit=API_CONCURRENCY_LIMIT 
    )
    init_end_time = time.time()
    print(f"\n--- [计时] BM25及意图通路框架初始化耗时: {init_end_time - init_start_time:.2f} 秒 ---\n")

    # --- 3. 计算所有分数 (统一流程) ---
    print("\n--- 步骤 3: 计算所有召回分数 ---")
    bm25_start_time = time.time()
    all_bm25_scores = [bm25_retriever.retrieve_scores(row['plan（在xx中做什么）']) for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="计算BM25分数")]
    # all_bm25_scores = [bm25_retriever.retrieve_scores(row['query']) for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="计算BM25分数")]
    
    bm25_end_time = time.time()

    semantic_start_time = time.time()
    all_plan_queries = data_df['plan（在xx中做什么）'].tolist()
    # all_plan_queries = data_df['query'].tolist()
    all_semantic_scores = await instruction_searcher.initialize_and_get_all_scores(all_plan_queries)
    semantic_end_time = time.time()
    
    total_queries = len(data_df)
    avg_bm25_latency = (bm25_end_time - bm25_start_time) / total_queries * 1000
    avg_semantic_latency = (semantic_end_time - semantic_start_time) / total_queries * 1000
    
    print(f"\n--- [计算时延分析] ---")
    print(f"  BM25通路平均时延: {avg_bm25_latency:.4f} 毫秒/查询")
    print(f"  精准意图通路平均时延 (模式: {MODE.upper()}): {avg_semantic_latency:.4f} 毫秒/查询")
    print("-" * 30)

    # --- 4. Alpha值网格搜索 ---
    print("\n--- 步骤 4: 开始进行Alpha值网格搜索 ---")
    alpha_range = np.linspace(0, 1, 101)
    best_alpha = -1
    best_score = -1
    
    for alpha in tqdm(alpha_range, desc="Alpha网格搜索中"):
        current_score = evaluate_recall_system(data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, K_VALUES)
        if current_score > best_score:
            best_score = current_score
            best_alpha = alpha

    print("\n--- Alpha值网格搜索完成 ---")
    print(f"找到的最佳Alpha值: {best_alpha:.2f} (对应的最高平均Recall@1为: {best_score:.4f})")
    # best_alpha = 1
    
    # --- 5. 使用最佳Alpha进行最终的、完整的评测 ---
    print(f"\n--- 步骤 5: 使用最佳Alpha={best_alpha:.2f}进行最终的完整评测 ---")
    results, error_cases, latency_records, detailed_predictions = evaluate_recall_system(
        data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, best_alpha, K_VALUES, full_report=True
    )
    
    # --- 6. 汇总并报告最终结果 ---
    print(f"\n\n--- 步骤 6: 最终评测结果报告 (模式: {MODE.upper()}, Alpha: {best_alpha:.2f}) ---")
    final_scores_report = {}
    for metric, vals in results.items():
        if metric == 'AUC': 
            final_scores_report['AUC'] = np.mean(vals['all'])
        else: 
            final_scores_report[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    
    report_df = pd.DataFrame({ m: final_scores_report[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    
    average_latency_ms = np.mean(latency_records) * 1000

    print(f"混合召回模型 (BM25 + 精准意图[{MODE.upper()}]) 在完整数据集上的评测结果:")
    print("-" * 70)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores_report['AUC']:.4f}")
    print(f"**平均查询处理时延 (分数融合+排序)**: {average_latency_ms:.4f} 毫秒/查询")
    print("-" * 70)

    # --- 7. 错误分析 和 8. 保存文件 ... 
    print(f"\n\n--- 步骤 7: Top-1 错误案例分析 (共 {len(error_cases)} 个错误) ---")
    if not error_cases:
        print("🎉 恭喜！在数据集上没有发现 Top-1 错误案例！")
    else:
        for i, case in enumerate(error_cases[:NUM_ERROR_EXAMPLES_TO_PRINT]):
            print(f"\n--- 错误案例 {i+1}/{len(error_cases)} ---")
            print(f"  [查询 Query]: {case['Query']}")
            print(f"  [真实工具 Ground Truth]: {case['Ground Truth']}")
            print(f"  [预测工具 Prediction@1]: {case['Prediction@1']}")
            print(f"  [预测工具 Prediction@5]: {case['Prediction@5']}")
        if len(error_cases) > NUM_ERROR_EXAMPLES_TO_PRINT:
            print(f"\n... (仅显示前 {NUM_ERROR_EXAMPLES_TO_PRINT} 个错误案例) ...")
    print("-" * 70)

    # --- 步骤 8: 保存召回结果到文件 ---
    print(f"\n\n--- 步骤 8: 保存详细召回结果到文件 ---")
    try:
        output_records = []
        for pred in detailed_predictions:
            record = {
                'query': pred['query'],
                'plan': pred['plan'],
                'ground_truth': ', '.join(list(pred['ground_truth'][0])) if pred['ground_truth'] else '',
            }
            for i, tool_info in enumerate(pred['retrieved_top_k']):
                record[f'pred_tool_{i+1}'] = tool_info['tool']
                record[f'pred_score_{i+1}'] = tool_info['score']
            output_records.append(record)
        
        output_df = pd.DataFrame(output_records)
        
        output_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 召回结果已成功保存到: {OUTPUT_FILE_PATH}")

    except Exception as e:
        print(f"❌ 保存召回结果失败: {e}")
    print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())