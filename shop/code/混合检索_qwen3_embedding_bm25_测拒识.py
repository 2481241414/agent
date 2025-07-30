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
# 区域 1: 检查并导入所需库
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer
    import torch
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install faiss-cpu torch sentence-transformers transformers rank_bm25 scikit-learn pandas tqdm")
    exit()

# ==============================================================================
# 区域 2: 召回器类定义
# ==============================================================================

class BM25Retriever:
    """【关键词通路】BM25召回器，专注于关键词和用户多样化表达的匹配。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, k1=1.5, b=0.75):
        """
        初始化BM25召回器。
        这个过程会构建一个基于关键词的索引，该索引将每个工具与一组相关的用户查询指令关联起来。

        Args:
            data_df (pd.DataFrame): 包含'指令'和'ground_truth_tool'列的标注数据。
            all_tools_definitions (list): 所有可用工具的定义列表。
            k1 (float, optional): BM25算法的参数k1。默认为1.5。
            b (float, optional): BM25算法的参数b。默认为0.75。
        """
        self.definitions = all_tools_definitions
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(all_tools_definitions)}
        self._add_jieba_words()
        
        print("--- [BM25通路] 正在构建关键词增强语料库... ---")
        corpus = self._build_keyword_rich_corpus(data_df)
        tokenized_corpus = [jieba.lcut(doc, cut_all=False) for doc in tqdm(corpus, desc="BM25语料库分词")]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        print("--- [BM25通路] 召回器构建完成 ---")

    def _add_jieba_words(self):
        """
        向Jieba分词器添加自定义词典。
        这包括所有工具的名称和一些领域内的核心词汇，以确保它们在分词时被视为一个整体，提高匹配准确率。
        """
        for tool in self.definitions:
            jieba.add_word(tool.get('name', '').split('(')[0], freq=100)
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "发票", "优惠券"]
        for word in core_words:
            jieba.add_word(word, freq=100)

    def _build_keyword_rich_corpus(self, data_df: pd.DataFrame) -> list:
        """
        构建一个为BM25优化的“关键词丰富”语料库。
        语料库中的每个文档对应一个工具。该文档由所有指向此工具的、去重后的用户'指令'文本拼接而成。
        
        Args:
            data_df (pd.DataFrame): 包含'指令'和'ground_truth_tool'列的标注数据。

        Returns:
            list: 一个字符串列表，每个字符串是对应工具的关键词文档。
        """
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
        """
        针对给定的查询，计算其与所有工具文档的BM25分数。

        Args:
            query (str): 用户的输入查询（通常是'plan'）。

        Returns:
            np.ndarray: 一个数组，包含查询与每个工具的BM25相似度分数。
        """
        tokenized_query = jieba.lcut(query, cut_all=False)
        return self.bm25.get_scores(tokenized_query)


class InstructionSearcher:
    """【精准意图通路】使用Qwen3+Faiss在“指令”空间中进行语义搜索。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, model_name: str):
        """
        初始化精准意图搜索器。
        这个过程会加载一个语义模型，将数据集中所有唯一的'指令'编码为向量，并构建一个Faiss索引用于快速的向量相似度搜索。

        Args:
            data_df (pd.DataFrame): 包含'指令'和'ground_truth_tool'列的标注数据。
            all_tools_definitions (list): 所有可用工具的定义列表。
            model_name (str): SentenceTransformer模型的名称或本地路径。
        """
        self.definitions = all_tools_definitions
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- [意图通路] 正在使用设备: {self.device} ---")
        
        self._build_mappings(data_df)
        
        print(f"--- [意图通路] 正在加载语义模型: {model_name} ---")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        
        print(f"--- [意图通路] 正在将 {len(self.unique_instructions)} 条唯一指令编码为向量... ---")
        instruction_embeddings = self.model.encode(self.unique_instructions, convert_to_tensor=False, show_progress_bar=True)
        self._build_faiss_index(instruction_embeddings)
        print("--- [意图通路] Qwen3+Faiss召回器准备就绪 ---\n")

    def _build_mappings(self, data_df: pd.DataFrame):
        """
        构建从唯一'指令'到其对应工具的映射关系。
        同时，创建工具名称到其在全局工具列表中索引的映射。

        Args:
            data_df (pd.DataFrame): 标注数据。
        """
        self.instruction_to_tool_map = {}
        for _, row in data_df.drop_duplicates(subset=['指令'], keep='last').iterrows():
            if pd.notna(row['指令']) and isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
                self.instruction_to_tool_map[row['指令']] = row['ground_truth_tool'][0]
        
        self.unique_instructions = list(self.instruction_to_tool_map.keys())
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(self.definitions)}

    def _build_faiss_index(self, embeddings: np.ndarray):
        """
        使用给定的向量构建一个Faiss索引。
        这里使用IndexFlatIP（内积）作为相似度度量，并对向量进行L2归一化，使得内积等价于余弦相似度。

        Args:
            embeddings (np.ndarray): 从'指令'文本编码而来的向量矩阵。
        """
        embeddings = embeddings.astype('float32')
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def retrieve_scores(self, plan_query: str) -> np.ndarray:
        """
        对给定的查询进行语义搜索，并返回所有工具的分数。
        过程：编码查询 -> 在Faiss中搜索最相似的K个'指令' -> 将这些'指令'的相似度分数传递给它们对应的工具。

        Args:
            plan_query (str): 用户的输入查询（通常是'plan'）。

        Returns:
            np.ndarray: 一个数组，包含查询与每个工具的最高语义相似度分数。
        """
        query_embedding = self.model.encode(plan_query, convert_to_tensor=False)
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
                        
        return tool_scores

# ==============================================================================
# 区域 3: 评测函数
# ==============================================================================
def _get_tool_names(tools: list) -> set:
    """
    一个辅助函数，从工具定义列表（例如ground truth或retrieved列表）中提取所有工具的名称。

    Args:
        tools (list): 一个包含工具定义字典的列表。

    Returns:
        set: 一个包含所有工具名称的集合，便于快速查找。
    """
    if not isinstance(tools, list): return set()
    return {tool.get('name') for tool in tools}

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    """
    计算Recall@K指标。
    衡量在前K个召回结果中，命中了多少比例的真实相关工具。

    Args:
        retrieved (list): 按分数排序的召回工具列表。
        ground_truth (list): 真实的工具列表。
        k (int): 截断位置。

    Returns:
        float: Recall@K的分数 (0.0到1.0)。
    """
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    return len(retrieved_names_at_k.intersection(ground_truth_names)) / len(ground_truth_names)

def calculate_completeness_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    """
    计算Completeness@K（完备率）指标。
    衡量在前K个召回结果中，是否召回了**所有**真实相关的工具。这是一个比Recall更严格的指标。

    Args:
        retrieved (list): 按分数排序的召回工具列表。
        ground_truth (list): 真实的工具列表。
        k (int): 截断位置。

    Returns:
        float: 如果所有真实工具都被召回则为1.0，否则为0.0。
    """
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    return 1.0 if ground_truth_names.issubset(retrieved_names_at_k) else 0.0

def calculate_ndcg_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    """
    计算NDCG@K (Normalized Discounted Cumulative Gain) 指标。
    衡量排序质量，越相关的结果排在越前面，得分越高。

    Args:
        retrieved (list): 按分数排序的召回工具列表。
        ground_truth (list): 真实的工具列表。
        k (int): 截断位置。

    Returns:
        float: NDCG@K的分数 (0.0到1.0)。
    """
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, tool in enumerate(retrieved[:k]) if tool.get('name') in ground_truth_names)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth_names), k)))
    return dcg / idcg if idcg > 0 else 0.0

def calculate_hit_ratio_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    """
    计算Hit Ratio@K (命中率) 指标。
    衡量在前K个召回结果中，是否**至少命中一个**真实相关的工具。

    Args:
        retrieved (list): 按分数排序的召回工具列表。
        ground_truth (list): 真实的工具列表。
        k (int): 截断位置。

    Returns:
        float: 如果至少命中一个则为1.0，否则为0.0。
    """
    retrieved_names = _get_tool_names(retrieved[:k])
    return 1.0 if retrieved_names & _get_tool_names(ground_truth) else 0.0

def calculate_average_precision_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    """
    计算Average Precision@K (AP@K) 指标。
    MAP@K (Mean Average Precision)的基础，综合考虑了排序位置和召回率。

    Args:
        retrieved (list): 按分数排序的召回工具列表。
        ground_truth (list): 真实的工具列表。
        k (int): 截断位置。

    Returns:
        float: AP@K的分数。
    """
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
    """
    计算Mean Reciprocal Rank@K (MRR@K) 指标。
    衡量第一个正确答案被排在第几位。其值为第一个正确答案排名的倒数。

    Args:
        retrieved (list): 按分数排序的召回工具列表。
        ground_truth (list): 真实的工具列表。
        k (int): 截断位置。

    Returns:
        float: MRR@K的分数。
    """
    gt_names = _get_tool_names(ground_truth)
    for i, tool in enumerate(retrieved[:k]):
        if tool.get('name') in gt_names:
            return 1.0 / (i + 1)
    return 0.0

def calculate_auc_for_query(all_scores: np.ndarray, tool_defs: list, ground_truth: list) -> float:
    """
    为单次查询计算AUC (Area Under the ROC Curve) 分数。
    衡量模型将正样本（相关工具）排在负样本（不相关工具）前面的整体能力。

    Args:
        all_scores (np.ndarray): 模型对所有工具给出的分数数组。
        tool_defs (list): 全局工具定义列表。
        ground_truth (list): 真实的工具列表。

    Returns:
        float: AUC分数。如果标签只有一类，则返回0.5。
    """
    gt_names = _get_tool_names(ground_truth)
    labels = [1 if t['name'] in gt_names else 0 for t in tool_defs]
    try:
        if len(set(labels)) < 2: return 0.5
        return roc_auc_score(labels, all_scores)
    except ValueError: return 0.5

# ==============================================================================
# 区域 4: 工具定义
# ==============================================================================
def get_exact_tool_definitions() -> list:
    """
    定义并返回所有可供系统召回的工具列表。
    这是系统中所有工具的“知识库”，是硬编码的。
    如果需要增删改工具，应直接修改此函数。

    Returns:
        list: 一个包含多个字典的列表，每个字典代表一个工具，包含'name'和'description'。
    """
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
# 区域 5: 评测核心逻辑 
# ==============================================================================
def evaluate_recall_system(data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, k_values, full_report=False):
    """
    对整个数据集进行一次完整的评测。
    此函数会遍历数据集中的每一行，融合BM25和语义分数，计算各项评测指标。

    Args:
        data_df (pd.DataFrame): 完整的评测数据集。
        all_bm25_scores (list): 预计算好的所有查询的BM25分数列表。
        all_semantic_scores (list): 预计算好的所有查询的语义分数列表。
        all_tools_definitions (list): 全局工具定义列表。
        alpha (float): BM25分数的权重，范围[0, 1]。语义分数的权重为(1-alpha)。
        k_values (list): 一个包含多个k值的列表 (例如 [1, 3, 5])，用于计算@K指标。
        full_report (bool, optional): 控制返回内容的详细程度。
            - 如果为 True，返回详细的评测结果、错误案例、时延记录和预测详情。
            - 如果为 False（默认），仅返回平均Recall@1分数，用于快速的网格搜索。

    Returns:
        - if full_report: (results, error_cases, latency_records, detailed_predictions)
        - if not full_report: float (mean Recall@1 score)
    """
    results = defaultdict(lambda: defaultdict(list))
    error_cases = []
    latency_records = []
    detailed_predictions = [] 

    # 内部嵌套函数，用于分数归一化
    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if (max_s - min_s) == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    for i, (_, row) in enumerate(data_df.iterrows()):
        start_time = time.time()

        ground_truth = row['ground_truth_tool']
        bm25_scores = all_bm25_scores[i]
        semantic_scores = all_semantic_scores[i]
        
        # 归一化与融合
        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)
        final_scores = alpha * norm_bm25 + (1 - alpha) * norm_semantic
        
        # 排序并获取召回结果
        sorted_indices = np.argsort(final_scores)[::-1]
        retrieved = [all_tools_definitions[i] for i in sorted_indices]
        retrieved_scores = final_scores[sorted_indices]
        
        end_time = time.time()
        latency_records.append(end_time - start_time)

        if full_report:
            # 记录详细的预测信息
            prediction_record = {
                "query": row['query'],
                "plan": row['plan（在xx中做什么）'],
                "ground_truth": [_get_tool_names(ground_truth)],
                "retrieved_top_k": [{"tool": t.get('name'), "score": float(s)} for t, s in zip(retrieved[:max(k_values)], retrieved_scores[:max(k_values)])]
            }
            detailed_predictions.append(prediction_record)

            # 计算所有指标
            for k in k_values:
                results['Recall@K'][k].append(calculate_recall_at_k(retrieved, ground_truth, k))
                results['HR@K'][k].append(calculate_hit_ratio_at_k(retrieved, ground_truth, k))
                results['MAP@K'][k].append(calculate_average_precision_at_k(retrieved, ground_truth, k))
                results['MRR@K'][k].append(calculate_mrr_at_k(retrieved, ground_truth, k))
                results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved, ground_truth, k))
                results['COMP@K'][k].append(calculate_completeness_at_k(retrieved, ground_truth, k))
            results['AUC']['all'].append(calculate_auc_for_query(final_scores, all_tools_definitions, ground_truth))
            
            # 记录错误案例
            if calculate_recall_at_k(retrieved, ground_truth, 1) < 1.0:
                gt_name = _get_tool_names(ground_truth).pop() if ground_truth else "N/A"
                pred_name_top1 = retrieved[0].get('name') if retrieved else "N/A"
                error_cases.append({"Query": row['plan（在xx中做什么）'], "Ground Truth": [gt_name], "Prediction@1": [pred_name_top1], "Prediction@5": [r.get('name') for r in retrieved[:5]]})
        else:
            # 快速模式，只计算Recall@1
            results['Recall@K'][1].append(calculate_recall_at_k(retrieved, ground_truth, 1))

    if full_report:
        return results, error_cases, latency_records, detailed_predictions
    else:
        return np.mean(results['Recall@K'][1])

# ==============================================================================
# 区域 6: 主程序 (集成网格搜索) 
# ==============================================================================
def main():
    """
    主执行函数。
    该函数串联了整个项目的流程：
    1. 配置参数和路径。
    2. 加载和预处理数据。
    3. 初始化两个召回器（BM25 和 InstructionSearcher）。
    4. 预计算所有查询的分数以加速后续步骤。
    5. 通过网格搜索寻找最佳的融合权重alpha。
    6. 使用最佳alpha进行一次完整的、详细的评测。
    7. 生成并打印最终的性能报告和错误案例分析。
    8. 将详细的召回结果保存到CSV文件。
    9. 【新增】对指定的拒识指令进行专项测试。
    """
    # --- 0. 配置区域 ---
    # 注意：请确保此处的路径在你的环境中是正确的
    annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_plan.csv'
    SEMANTIC_MODEL_NAME = '/home/workspace/lgq/shop/model/Qwen3-Embedding-0.6B'
    K_VALUES = [1, 2, 3, 5, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10
    OUTPUT_FILE_PATH = '/home/workspace/lgq/shop/data/hybrid_recall_results.csv' 

    # --- 1. 数据加载 ---
    print("--- 步骤 1: 加载完整数据集 ---")
    try:
        # 注意: 你的拒识数据没有 '指令' 列，为了让主流程能跑通，我们只读取公共的列
        # 在评测时，BM25和InstructionSearcher的构建仍依赖于这个有'指令'列的原始文件
        required_columns = ['query', 'plan（在xx中做什么）', 'ground_truth_tool', '指令']
        data_df = pd.read_csv(annotated_data_file_path, usecols=lambda c: c in required_columns).dropna(subset=['plan（在xx中做什么）']).reset_index(drop=True)
        # 内部嵌套函数，用于安全地解析字符串格式的列表
        def parse_tools(s): return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else []
        data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(parse_tools)
        print(f"数据加载完成: 共 {len(data_df)} 条。\n")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到，请检查路径 '{annotated_data_file_path}'")
        return
    except Exception as e:
        print(f"错误: 读取或解析文件 '{annotated_data_file_path}' 失败. {e}")
        return

    # --- 2. 初始化双路召回器 ---
    all_tools_definitions = get_exact_tool_definitions()
    
    init_start_time = time.time()
    # 注意: 召回器的构建是基于你的原始训练数据的
    bm25_retriever = BM25Retriever(data_df, all_tools_definitions)
    instruction_searcher = InstructionSearcher(data_df, all_tools_definitions, model_name=SEMANTIC_MODEL_NAME)
    init_end_time = time.time()
    print(f"\n--- [计时] 初始化所有召回器总耗时: {init_end_time - init_start_time:.2f} 秒 ---\n")

    
    # --- 3. 计算所有分数以加速网格搜索 ---
    print("\n--- 步骤 3: 计算所有召回分数以加速网格搜索 ---")
    bm25_start_time = time.time()
    all_bm25_scores = [bm25_retriever.retrieve_scores(row['plan（在xx中做什么）']) for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="计算BM25分数")]
    bm25_end_time = time.time()

    semantic_start_time = time.time()
    all_semantic_scores = [instruction_searcher.retrieve_scores(row['plan（在xx中做什么）']) for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="计算语义分数")]
    semantic_end_time = time.time()
    
    total_queries = len(data_df)
    avg_bm25_latency = (bm25_end_time - bm25_start_time) / total_queries * 1000
    avg_semantic_latency = (semantic_end_time - semantic_start_time) / total_queries * 1000
    
    print(f"\n--- [计算时延分析] ---")
    print(f"  BM25通路平均时延: {avg_bm25_latency:.4f} 毫秒/查询")
    print(f"  精准意图通路平均时延: {avg_semantic_latency:.4f} 毫秒/查询")
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
    
    # --- 5. 使用最佳Alpha进行最终的、完整的评测 ---
    print(f"\n--- 步骤 5: 使用最佳Alpha={best_alpha:.2f}进行最终的完整评测 ---")
    results, error_cases, latency_records, detailed_predictions = evaluate_recall_system(
        data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, best_alpha, K_VALUES, full_report=True
    )
    
    # --- 6. 汇总并报告最终结果 ---
    print("\n\n--- 步骤 6: 最终评测结果报告 (使用最佳Alpha) ---")
    final_scores_report = {}
    for metric, vals in results.items():
        if metric == 'AUC': 
            final_scores_report['AUC'] = np.mean(vals['all'])
        else: 
            final_scores_report[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    
    report_df = pd.DataFrame({ m: final_scores_report[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    
    average_latency_ms = np.mean(latency_records) * 1000

    print("混合召回模型 (BM25 + 精准意图) 在完整数据集上的评测结果:")
    print("-" * 70)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores_report['AUC']:.4f}")
    print(f"**平均查询处理时延 (分数融合+排序)**: {average_latency_ms:.4f} 毫秒/查询")
    print("-" * 70)
    
    # --- 7. 打印错误分析报告 ---
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
    
    # --- 步骤 9: 拒识能力专项测试 ---
    # 定义你想要测试的拒识数据集
    rejection_test_data = {
        'query': [
            '放一首晴天',
            '用滴滴帮我叫个快车',
            '嘿Siri，帮我看一下外面天气怎么样',
            '帮我订张去南京的机票',
            '打开高德，我要导航去景德镇玩'
        ],
        'plan（在xx中做什么）': [
            '在$$中播放晴天',
            '在$滴滴$中叫个快车',
            '在$$中查天气',
            '在$$中订一张去南京的机票',
            '在$高德$中导航到景德镇'
        ]
    }
    rejection_df = pd.DataFrame(rejection_test_data)
    queries_for_rejection_test = rejection_df['plan（在xx中做什么）'].tolist()

    # 调用测试函数
    test_rejection_queries(
        queries_to_test=queries_for_rejection_test,
        bm25_retriever=bm25_retriever,
        instruction_searcher=instruction_searcher,
        all_tools_definitions=all_tools_definitions,
        alpha=best_alpha, # 使用前面找到的最佳alpha
        top_k=5
    )


# ==============================================================================
# 【新增】区域 7: 拒识能力专项测试函数
# ==============================================================================
def test_rejection_queries(queries_to_test: list, bm25_retriever: BM25Retriever, instruction_searcher: InstructionSearcher, all_tools_definitions: list, alpha: float, top_k: int = 5):
    """
    对一组给定的“拒识”查询进行测试，并打印出每个查询的Top-K召回结果。
    这个函数专门用于检验模型对于领域外指令的反应。

    Args:
        queries_to_test (list): 一个包含领域外查询字符串的列表 (通常是'plan'列的内容)。
        bm25_retriever (BM25Retriever): 已经初始化并训练好的BM25召回器。
        instruction_searcher (InstructionSearcher): 已经初始化并加载好索引的意图搜索器。
        all_tools_definitions (list): 全局工具定义列表。
        alpha (float): 用于融合分数的最佳权重。
        top_k (int, optional): 为每个查询显示的最高分结果数量。默认为5。
    """
    print(f"\n\n--- 步骤 9: 开始进行拒识能力专项测试 ---")
    print("-" * 70)

    # 内部嵌套函数，用于分数归一化
    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if (max_s - min_s) == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    for i, query in enumerate(queries_to_test):
        if not query or pd.isna(query):
            print(f"\n--- 测试查询 {i+1} (已跳过, 输入为空) ---")
            continue

        print(f"\n--- 测试查询 {i+1}: \"{query}\" ---")

        # 1. 获取两路得分
        bm25_scores = bm25_retriever.retrieve_scores(query)
        semantic_scores = instruction_searcher.retrieve_scores(query)

        # 2. 归一化与融合
        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)
        final_scores = alpha * norm_bm25 + (1 - alpha) * norm_semantic

        # 3. 排序并获取Top-K结果
        sorted_indices = np.argsort(final_scores)[::-1]
        retrieved_tools = [all_tools_definitions[idx] for idx in sorted_indices[:top_k]]
        retrieved_scores = final_scores[sorted_indices[:top_k]]

        # 4. 打印结果
        print("模型召回的Top-5工具及其分数:")
        if not retrieved_tools:
            print("  -> 模型没有返回任何结果。")
        else:
            for rank, (tool, score) in enumerate(zip(retrieved_tools, retrieved_scores)):
                print(f"  Rank {rank+1}: {tool.get('name', 'N/A')} (Score: {score:.4f})")
    print("-" * 70)


if __name__ == "__main__":
    main()