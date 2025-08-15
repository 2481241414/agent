import pandas as pd
import os
import json
import ast
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
import time
import asyncio
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
    import aiohttp
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install faiss-cpu torch sentence-transformers transformers rank_bm25 scikit-learn pandas tqdm aiohttp")
    exit()

# ==============================================================================
# 区域 2: 召回器类定义
# ==============================================================================
class BM25Retriever:
    """【关键词通路】BM25召回器，语料库直接来自工具描述。"""
    def __init__(self, all_tools_definitions: list, k1=1.5, b=0.75):
        self.definitions = all_tools_definitions
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(all_tools_definitions)}
        self._add_jieba_words()
        
        print("--- [BM25通路] 正在基于【工具描述】构建关键词语料库... ---")
        corpus = self._build_corpus_from_descriptions()
        tokenized_corpus = [jieba.lcut(doc, cut_all=False) for doc in tqdm(corpus, desc="BM25语料库分词")]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        print("--- [BM25通路] 召回器构建完成 ---")

    def _add_jieba_words(self):
        for tool in self.definitions:
            jieba.add_word(tool['name'].split('(')[0], freq=100)
        for word in ["购物车", "待收货", "待付款", "收藏夹", "发票", "优惠券"]:
            jieba.add_word(word, freq=100)

    def _build_corpus_from_descriptions(self) -> list:
        corpus = [''] * len(self.definitions)
        for tool_def in self.definitions:
            tool_idx = self.tool_name_to_idx[tool_def['name']]
            # document = f"{tool_def['name'].split('(')[0].replace('_', ' ')} {tool_def.get('description', '')}"
            document = f"{tool_def.get('description', '')}"
            print(document)
            corpus[tool_idx] = document
        return corpus

    def retrieve_scores(self, query: str) -> np.ndarray:
        tokenized_query = jieba.lcut(query, cut_all=False)
        return self.bm25.get_scores(tokenized_query)

class DescriptionSearcher:
    """【语义通路】匹配用户Query与工具Description的语义相似度。"""
    def __init__(self, all_tools_definitions: list, mode: str, model_path_or_name: str, api_url: str = None, api_concurrency_limit: int = 10):
        self.definitions = all_tools_definitions
        self.mode = mode
        self.model_path_or_name = model_path_or_name
        self.api_url = api_url
        self.model = None
        self.faiss_index = None
        self.api_concurrency_limit = api_concurrency_limit

        print(f"--- [语义通路] 模式: {self.mode.upper()} ---")
        if self.mode == 'local':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"--- [语义通路] 正在使用设备: {self.device} ---")
            self.model = SentenceTransformer(self.model_path_or_name, trust_remote_code=True, device=self.device)
        
        self._build_mappings_from_descriptions()

    def _build_mappings_from_descriptions(self):
        self.description_to_tool_map = {tool.get('description', ''): tool for tool in self.definitions if tool.get('description')}
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
        async def fetch(text):
            if not text or text.isspace(): return None
            async with semaphore:
                headers = {"Content-Type": "application/json"}
                payload = {"model": self.model_path_or_name, "input": text}
                try:
                    async with session.post(self.api_url, headers=headers, json=payload, timeout=60) as resp:
                        if resp.status == 200:
                            return (await resp.json()).get('data', [{}])[0].get('embedding')
                        else:
                            print(f"API请求失败, 状态码: {resp.status}, 文本: '{text}'")
                            return None
                except Exception as e:
                    print(f"API请求异常: {e}, 文本: '{text}'")
                    return None
        return await asyncio.gather(*(fetch(text) for text in texts))

    async def initialize_and_get_all_scores(self, all_user_queries: list):
        print(f"--- [语义通路] 正在编码 {len(self.unique_descriptions)} 条【工具描述】...")
        if self.mode == 'local':
            desc_embs = self.model.encode(self.unique_descriptions, show_progress_bar=True)
        else:
            async with aiohttp.ClientSession() as session:
                desc_embs = await self._get_embeddings_from_api(self.unique_descriptions, session)
        
        valid_desc_embs = [(d, e) for d, e in zip(self.unique_descriptions, desc_embs) if e is not None]
        if not valid_desc_embs: raise RuntimeError("所有工具描述向量化失败!")
        
        valid_desc, valid_embs = zip(*valid_desc_embs)
        self.unique_descriptions = list(valid_desc)
        self.description_to_tool_map = {d: self.description_to_tool_map[d] for d in self.unique_descriptions}
            
        self._build_faiss_index(np.array(valid_embs))
        print("--- [语义通路] Faiss索引构建完成 ---")

        print(f"--- [语义通路] 正在对 {len(all_user_queries)} 条【用户Query】进行评分...")
        if self.mode == 'local':
            query_embs = self.model.encode(all_user_queries, show_progress_bar=True)
        else:
             async with aiohttp.ClientSession() as session:
                query_embs = await self._get_embeddings_from_api(all_user_queries, session)
        
        all_scores = []
        for q_emb in tqdm(query_embs, desc="计算语义分数 (query vs desc)"):
            if q_emb is None:
                all_scores.append(np.zeros(len(self.definitions)))
                continue
            q_emb_np = np.array([q_emb], dtype='float32')
            faiss.normalize_L2(q_emb_np)
            dists, idxs = self.faiss_index.search(q_emb_np, k=min(len(self.unique_descriptions), 50))
            
            tool_scores = np.zeros(len(self.definitions))
            for dist, idx in zip(dists[0], idxs[0]):
                if idx != -1:
                    tool_def = self.description_to_tool_map.get(self.unique_descriptions[idx])
                    if tool_def and (tool_idx := self.tool_name_to_idx.get(tool_def['name'])) is not None:
                        tool_scores[tool_idx] = max(tool_scores[tool_idx], dist)
            all_scores.append(tool_scores)
        return all_scores

# ==============================================================================
# 区域 3: 评测指标函数
# ==============================================================================
def _get_tool_names(tools: list) -> set:
    if not isinstance(tools, list): return set()
    return {tool.get('name') for tool in tools}

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
    return len(retrieved_names_at_k.intersection(ground_truth_names)) / len(ground_truth_names)

def calculate_completeness_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0
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
    hit_count, sum_prec = 0, 0.0
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
        return roc_auc_score(labels, all_scores) if len(set(labels)) > 1 else 0.5
    except ValueError: return 0.5

# ==============================================================================
# 区域 4: 工具定义
# ==============================================================================
def get_exact_tool_definitions():
    """重要：此函数返回的工具定义中，'name' 字段必须被标准化为 'function_name(...)' 格式。"""
    full_tools = [
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
    for tool in full_tools:
        tool['name'] = tool['name'].split('(')[0] + "(...)"
    return full_tools

# ==============================================================================
# 区域 5: 评测核心逻辑
# ==============================================================================
def evaluate_recall_system(data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, k_values, full_report=False):
    results = defaultdict(lambda: defaultdict(list))
    latency_records = []
    detailed_predictions = [] if full_report else None
    error_cases = [] if full_report else None

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        return np.zeros_like(scores) if (max_s - min_s) == 0 else (scores - min_s) / (max_s - min_s)

    score_indices = data_df.index.tolist()
    
    iterator = data_df.iterrows()
    if full_report:
        iterator = tqdm(iterator, total=len(data_df), desc="评测中")

    for i, (_, row) in enumerate(iterator):
        start_time = time.time()
        original_index = score_indices[i]
        ground_truth = row['ground_truth_tool']
        
        bm25_s = np.array(all_bm25_scores[original_index])
        semantic_s = np.array(all_semantic_scores[original_index])
        final_scores = alpha * normalize(bm25_s) + (1 - alpha) * normalize(semantic_s)
        
        sorted_indices = np.argsort(final_scores)[::-1]
        retrieved = [all_tools_definitions[idx] for idx in sorted_indices]
        latency_records.append(time.time() - start_time)

        if full_report:
            for k in k_values:
                results['Recall@K'][k].append(calculate_recall_at_k(retrieved, ground_truth, k))
                results['HR@K'][k].append(calculate_hit_ratio_at_k(retrieved, ground_truth, k))
                results['MAP@K'][k].append(calculate_average_precision_at_k(retrieved, ground_truth, k))
                results['MRR@K'][k].append(calculate_mrr_at_k(retrieved, ground_truth, k))
                results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved, ground_truth, k))
                results['COMP@K'][k].append(calculate_completeness_at_k(retrieved, ground_truth, k))
            results['AUC']['all'].append(calculate_auc_for_query(final_scores, all_tools_definitions, ground_truth))
            
            if calculate_completeness_at_k(retrieved, ground_truth, 1) < 1.0:
                error_cases.append({
                    "Query": row['query'], 
                    "Ground Truth": list(_get_tool_names(ground_truth)), 
                    "Prediction@1": [retrieved[0].get('name')] if retrieved else ["N/A"],
                    "Prediction@5": [r.get('name') for r in retrieved[:5]]
                })
        else:
            results['Recall@K'][1].append(calculate_recall_at_k(retrieved, ground_truth, 1))

    # 平均所有指标
    final_metrics = {}
    for metric, k_scores in results.items():
        if metric == 'AUC':
            final_metrics[metric] = np.mean(k_scores['all'])
        else:
            final_metrics[metric] = {k: np.mean(scores) for k, scores in k_scores.items()}

    if full_report:
        for i, (_, row) in enumerate(data_df.iterrows()):
             original_index = score_indices[i]
             final_scores = alpha * normalize(np.array(all_bm25_scores[original_index])) + \
                            (1 - alpha) * normalize(np.array(all_semantic_scores[original_index]))
             sorted_indices = np.argsort(final_scores)[::-1]
             retrieved = [all_tools_definitions[idx] for idx in sorted_indices]
             detailed_predictions.append({
                 "query": row['query'],
                 "ground_truth": ', '.join(_get_tool_names(row['ground_truth_tool'])),
                 **{f"pred_tool_{i+1}": t.get('name', '') for i, t in enumerate(retrieved[:max(k_values)])},
                 **{f"pred_score_{i+1}": float(s) for i, s in enumerate(final_scores[sorted_indices][:max(k_values)])}
             })
        return final_metrics, error_cases, latency_records, detailed_predictions
    else:
        return final_metrics.get('Recall@K', {}).get(1, 0.0)

# ==============================================================================
# 区域 6: 主程序
# ==============================================================================
def run_full_evaluation_on_subset(
    subset_df, subset_name, all_bm25_scores, all_semantic_scores, all_tools_definitions, 
    k_values, mode, num_error_examples, output_file_path=None
):
    if subset_df.empty:
        print(f"\n--- 数据子集 '{subset_name}' 为空，跳过评测。 ---")
        return

    print(f"\n\n{'='*30}\n--- 开始对【{subset_name}】子集 (共 {len(subset_df)} 条) 进行评测 ---\n{'='*30}")
    
    best_alpha, best_score = -1, -1
    for alpha in tqdm(np.linspace(0, 1, 51), desc=f"Alpha网格搜索 ({subset_name})"):
        score = evaluate_recall_system(subset_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, k_values)
        if score > best_score:
            best_score, best_alpha = score, alpha
    
    print(f"\n--- ({subset_name}) 最佳Alpha值: {best_alpha:.2f} (最高Recall@1: {best_score:.4f}) ---")
    
    final_metrics, error_cases, latency_records, detailed_predictions = evaluate_recall_system(
        subset_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, best_alpha, k_values, full_report=True
    )
    
    print(f"\n--- 最终评测结果报告 (子集: {subset_name}, Alpha: {best_alpha:.2f}) ---")
    
    report_df = pd.DataFrame(final_metrics).drop('AUC', errors='ignore').T
    report_df.columns = [f"@{k}" for k in report_df.columns]
    
    # 【修正点2】修改报告标题
    print(f"混合召回模型 (query-vs-description [{mode.upper()}]) 在【{subset_name}】数据集上的召回评估指标:")
    print("-" * 70)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_metrics.get('AUC', 0.0):.4f}")
    if latency_records:
        print(f"**平均查询处理时延 (分数融合+排序)**: {np.mean(latency_records) * 1000:.4f} 毫秒/查询")
    print("-" * 70)

    print(f"\n--- ({subset_name}) Top-1 错误案例分析 (共 {len(error_cases)} 个错误) ---")
    if not error_cases:
        print("🎉 恭喜！没有发现 Top-1 错误案例！")
    else:
        for i, case in enumerate(error_cases[:num_error_examples]):
            print(f"\n--- 错误案例 {i+1}/{len(error_cases)} ---")
            print(f"  [查询 Query]: {case['Query']}")
            print(f"  [真实工具 Ground Truth]: {case['Ground Truth']}")
            print(f"  [预测工具 Prediction@1]: {case['Prediction@1']}")
            print(f"  [预测工具 Prediction@5]: {case['Prediction@5']}")
    
    if output_file_path:
        pd.DataFrame(detailed_predictions).to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ [{subset_name}] 详细召回结果已保存到: {output_file_path}")

async def main():
    # --- 0. 配置区域 ---
    MODE = 'api' 
    VLLM_API_URL = "http://localhost:8000/v1/embeddings"
    VLLM_SERVED_MODEL_NAME = "/home/workspace/ms-swift/output/Qwen3-Embedding-0.6B-sft-desc/v2-20250811-090239/checkpoint-280"
    API_CONCURRENCY_LIMIT = 20
    
    annotated_data_file_path = '/home/workspace/lgq/shop/data/plan_0803_1931_single_embedding_test.csv'
    
    K_VALUES = [1, 2, 3, 5, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 5
    base_output_path = f'/home/workspace/lgq/shop/data/evaluate/recall_results_query_vs_desc_full_metrics_{MODE}'

    # --- 1. 加载和预处理数据集 ---
    print("--- 步骤 1: 加载和预处理数据集 ---")
    try:
        data_df = pd.read_csv(annotated_data_file_path, usecols=['tag', 'query', 'fc_gt'])
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: 无法加载或解析CSV文件 '{annotated_data_file_path}'.\n请确保文件存在且包含 'tag', 'query', 'fc_gt' 列。\n具体错误: {e}")
        return

    data_df.rename(columns={'fc_gt': 'ground_truth_tool'}, inplace=True)
    data_df = data_df.dropna(subset=['ground_truth_tool', 'query']).reset_index(drop=True)

    def parse_and_standardize_gt(s):
        if not isinstance(s, str): return []
        try:
            parsed_list = ast.literal_eval(s)
            return [{'name': item[1].split('(')[0] + "(...)"} for item in parsed_list]
        except (ValueError, SyntaxError, IndexError):
            return []
            
    data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(parse_and_standardize_gt)
    
    single_task_df = data_df[data_df['tag'] == '单任务'].copy()
    multi_task_df = data_df[data_df['tag'] == '多任务'].copy()
    print(f"数据加载完成: 共 {len(data_df)} 条 (单任务: {len(single_task_df)}, 多任务: {len(multi_task_df)})。\n")

    # --- 2. 初始化召回器 ---
    all_tools_definitions = get_exact_tool_definitions()
    bm25_retriever = BM25Retriever(all_tools_definitions)
    description_searcher = DescriptionSearcher(
        all_tools_definitions, MODE, VLLM_SERVED_MODEL_NAME, VLLM_API_URL, API_CONCURRENCY_LIMIT 
    )

    # --- 3. 一次性计算所有分数 ---
    print("\n--- 步骤 3: 为全量数据计算所有召回分数 ---")
    all_queries = data_df['query'].tolist()
    all_bm25_scores = [bm25_retriever.retrieve_scores(q) for q in tqdm(all_queries, desc="计算BM25分数")]
    all_semantic_scores = await description_searcher.initialize_and_get_all_scores(all_queries)
    
    # --- 4. 运行评测 ---
    for subset_df, subset_name in [(single_task_df, "单任务"), (multi_task_df, "多任务"), (data_df, "整体")]:
        run_full_evaluation_on_subset(
            subset_df, subset_name, all_bm25_scores, all_semantic_scores, all_tools_definitions,
            K_VALUES, MODE, NUM_ERROR_EXAMPLES_TO_PRINT, 
            output_file_path=f"{base_output_path}_{subset_name.lower()}.csv"
        )

if __name__ == "__main__":
    asyncio.run(main())