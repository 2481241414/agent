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
            if 'ground_truth_tool' in row and isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
                if 'instruction' in row and pd.notna(row['instruction']): # 确保使用 'instruction' 列
                    tool_name = row['ground_truth_tool'][0]['name']
                    tool_text_aggregator[tool_name].append(row['instruction'])

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
    """【精准意图通路】使用Qwen3+Faiss在“指令”空间中进行语义搜索。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, model_name: str):
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
        self.instruction_to_tool_map = {}
        # 确保使用有 'instruction' 和 'ground_truth_tool' 列的数据进行映射构建
        data_with_instructions = data_df.dropna(subset=['instruction', 'ground_truth_tool'])

        for _, row in data_with_instructions.drop_duplicates(subset=['instruction'], keep='last').iterrows():
            if isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
                self.instruction_to_tool_map[row['instruction']] = row['ground_truth_tool'][0]
        
        self.unique_instructions = list(self.instruction_to_tool_map.keys())
        self.tool_name_to_idx = {tool['name']: i for i, tool in enumerate(self.definitions)}

    def _build_faiss_index(self, embeddings: np.ndarray):
        embeddings = embeddings.astype('float32')
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def retrieve_scores(self, plan_query: str) -> np.ndarray:
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
def get_exact_tool_definitions() -> list:
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
    results = defaultdict(lambda: defaultdict(list))
    error_cases = []
    latency_records = []
    detailed_predictions = [] 

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if (max_s - min_s) == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    query_column = 'plan_out' if 'plan_out' in data_df.columns else 'plan（在xx中做什么）'
    
    for i, (_, row) in enumerate(data_df.iterrows()):
        start_time = time.time()

        ground_truth = row['ground_truth_tool']
        bm25_scores = all_bm25_scores[i]
        semantic_scores = all_semantic_scores[i]
        
        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)
        final_scores = alpha * norm_bm25 + (1 - alpha) * norm_semantic
        
        sorted_indices = np.argsort(final_scores)[::-1]
        retrieved = [all_tools_definitions[i] for i in sorted_indices]
        retrieved_scores = final_scores[sorted_indices]
        
        end_time = time.time()
        latency_records.append(end_time - start_time)

        if full_report:
            query_text = row.get(query_column, "N/A")
            original_query_text = row.get('org_user_query', query_text) 

            prediction_record = {
                "query": original_query_text,
                "plan": query_text,
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
                error_cases.append({"Query": query_text, "Ground Truth": [gt_name], "Prediction@1": [pred_name_top1], "Prediction@5": [r.get('name') for r in retrieved[:5]]})
        else:
            results['Recall@K'][1].append(calculate_recall_at_k(retrieved, ground_truth, 1))

    if full_report:
        return results, error_cases, latency_records, detailed_predictions
    else:
        return np.mean(results['Recall@K'][1])

# ==============================================================================
# 区域 6: 主程序
# ==============================================================================
def main():
    # --- 0. 配置区域 ---
    base_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_plan.csv'
    new_format_test_file_path = '/home/workspace/lgq/shop/data/FC能力验证数据集 - 修正-0726.csv' # <--- ！！在这里替换为你的新数据文件名！！

    SEMANTIC_MODEL_NAME = '/home/workspace/lgq/shop/model/Qwen3-Embedding-0.6B'
    K_VALUES = [1, 2, 3, 5, 10]
    
    # --- 1. 数据加载 (用于构建召回器) ---
    print("--- 步骤 1: 加载用于构建召回器的基础数据集 ---")
    try:
        # 在旧格式中，列名可能是 '指令' 或 'instruction'
        # 我们同时加载，以便后续统一处理
        base_df = pd.read_csv(base_data_file_path)
        # 统一列名：如果存在 '指令'，就重命名为 'instruction'
        if '指令' in base_df.columns:
            base_df.rename(columns={'指令': 'instruction'}, inplace=True)
        
        # 确保基础数据中有 'instruction' 和 'ground_truth_tool'
        if 'instruction' not in base_df.columns or 'ground_truth_tool' not in base_df.columns:
             raise ValueError("基础数据文件必须包含 'instruction' 和 'ground_truth_tool' 列")

        def parse_tools(s): return ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else []
        base_df['ground_truth_tool'] = base_df['ground_truth_tool'].apply(parse_tools)
        
        # 筛选出有效行
        base_df = base_df.dropna(subset=['instruction', 'ground_truth_tool']).reset_index(drop=True)
        print(f"基础数据加载并清洗完成: 共 {len(base_df)} 条有效数据。\n")

    except FileNotFoundError:
        print(f"错误: 基础数据文件未找到，请检查路径 '{base_data_file_path}'")
        return
    except Exception as e:
        print(f"错误: 读取或解析基础文件 '{base_data_file_path}' 失败. {e}")
        return

    # --- 2. 初始化双路召回器 ---
    all_tools_definitions = get_exact_tool_definitions()
    
    init_start_time = time.time()
    bm25_retriever = BM25Retriever(base_df, all_tools_definitions)
    instruction_searcher = InstructionSearcher(base_df, all_tools_definitions, model_name=SEMANTIC_MODEL_NAME)
    init_end_time = time.time()
    print(f"\n--- [计时] 初始化所有召回器总耗时: {init_end_time - init_start_time:.2f} 秒 ---\n")

    # --- 3. 寻找最佳Alpha (使用原始数据集) ---
    print("\n--- 步骤 3: 使用基础数据集寻找最佳Alpha ---")
    # 确定原始数据集的查询列
    query_col_base = 'plan（在xx中做什么）'
    if query_col_base not in base_df.columns:
        print(f"警告: 基础数据中未找到 '{query_col_base}' 列，跳过Alpha搜索。将使用默认alpha=0.5")
        best_alpha = 0.5
    else:
        eval_df = base_df.dropna(subset=[query_col_base]).copy()
        print("--> 正在为基础数据集计算召回分数...")
        all_bm25_scores = [bm25_retriever.retrieve_scores(row[query_col_base]) for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="计算BM25分数")]
        all_semantic_scores = [instruction_searcher.retrieve_scores(row[query_col_base]) for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="计算语义分数")]
        
        print("--> 正在进行Alpha值网格搜索...")
        alpha_range = np.linspace(0, 1, 21)
        best_alpha = -1
        best_score = -1
        
        for alpha in tqdm(alpha_range, desc="Alpha网格搜索中"):
            current_score = evaluate_recall_system(eval_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, K_VALUES)
            if current_score > best_score:
                best_score = current_score
                best_alpha = alpha
        print(f"\n--> 网格搜索完成。找到的最佳Alpha值: {best_alpha:.2f} (最高平均Recall@1: {best_score:.4f})")

    print("-" * 70)

    # --- 步骤 4: 针对新格式数据集进行专项评测 ---
    run_evaluation_on_new_dataset(
        test_file_path=new_format_test_file_path,
        base_df_for_mapping=base_df, # 传入用于创建映射表的基础数据
        bm25_retriever=bm25_retriever,
        instruction_searcher=instruction_searcher,
        all_tools_definitions=all_tools_definitions,
        alpha=best_alpha,
        k_values=K_VALUES
    )

# ==============================================================================
# 【修改后】区域 7: 新格式数据集评测函数
# ==============================================================================
def create_instruction_to_tool_map(base_df: pd.DataFrame) -> dict:
    """
    根据基础数据集，创建一个从'instruction'到其对应工具定义的权威映射。
    """
    instruction_map = {}
    # 使用去重后的数据确保每个指令只映射到一个工具
    for _, row in base_df.drop_duplicates(subset=['instruction'], keep='last').iterrows():
        if isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
            instruction_map[row['instruction']] = row['ground_truth_tool']
    print(f"--- 已从基础数据中成功创建 'instruction' -> 'tool' 映射表，共 {len(instruction_map)} 条映射。 ---")
    return instruction_map

def run_evaluation_on_new_dataset(test_file_path: str, base_df_for_mapping: pd.DataFrame, bm25_retriever: BM25Retriever, instruction_searcher: InstructionSearcher, all_tools_definitions: list, alpha: float, k_values: list):
    """
    加载、筛选、处理并评测一个新格式的数据集。
    Ground Truth 将通过 'instruction' 列和映射表来确定。
    """
    print(f"\n\n--- 步骤 4: 开始针对新格式数据集进行专项评测 ---")
    print(f"--- 数据源: {test_file_path} ---")
    print("-" * 70)

    # --- 1. 创建 Instruction -> Tool 的映射表 ---
    instruction_tool_map = create_instruction_to_tool_map(base_df_for_mapping)

    # --- 2. 加载并筛选测试数据 ---
    try:
        # 【修改】现在需要'instruction'列来查找GT
        required_cols = ['tag', 'plan_out', 'instruction', 'org_user_query']
        test_df = pd.read_csv(test_file_path, usecols=lambda c: c in required_cols)
        
        tags_to_include = ['单指令单工具-常规', '单指令单工具-多槽位', '单指令单工具-工具多传']
        
        # 【修改】筛选条件现在是'plan_out'和'instruction'
        filtered_df = test_df[test_df['tag'].isin(tags_to_include)].dropna(subset=['plan_out', 'instruction']).reset_index(drop=True)
        
        if len(filtered_df) == 0:
            print("错误: 根据tag筛选后，没有可用于评测的数据。请检查CSV文件内容、tag值以及'plan_out'和'instruction'列。")
            return
            
        print(f"测试数据加载完成。根据tag筛选后，共获得 {len(filtered_df)} 条测试数据。")

    except FileNotFoundError:
        print(f"错误: 专项测试文件未找到，请检查路径 '{test_file_path}'")
        return
    except Exception as e:
        print(f"错误: 读取或解析专项测试文件 '{test_file_path}' 失败. {e}")
        return

    # --- 3. 【修改】通过映射表生成Ground Truth ---
    # 使用'instruction'列去映射表中查找对应的工具定义
    filtered_df['ground_truth_tool'] = filtered_df['instruction'].map(instruction_tool_map)
    
    # 清理那些在映射表中找不到对应工具的数据行
    initial_count = len(filtered_df)
    filtered_df.dropna(subset=['ground_truth_tool'], inplace=True)
    final_count = len(filtered_df)
    
    if initial_count > final_count:
        print(f"警告: {initial_count - final_count} 行数据因其 'instruction' 值在映射表中未找到而被丢弃。")
    
    if final_count == 0:
        print("错误: 所有测试数据的'instruction'都无法在基础数据中找到匹配，无法进行评测。")
        return

    # --- 4. 预计算所有分数 ---
    print("\n--- 正在为专项测试集计算召回分数 ---")
    all_bm25_scores = [bm25_retriever.retrieve_scores(row['plan_out']) for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="计算BM25分数")]
    all_semantic_scores = [instruction_searcher.retrieve_scores(row['plan_out']) for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="计算语义分数")]

    # --- 5. 执行完整评测 ---
    print(f"\n--- 使用Alpha={alpha:.2f}进行专项评测 ---")
    results, error_cases, latency_records, _ = evaluate_recall_system(
        filtered_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, k_values, full_report=True
    )
    
    # --- 6. 汇总并报告结果 ---
    print("\n\n--- 专项测试最终评测结果报告 ---")
    final_scores_report = {}
    for metric, vals in results.items():
        if metric == 'AUC': 
            final_scores_report['AUC'] = np.mean(vals['all'])
        else: 
            final_scores_report[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    
    report_df = pd.DataFrame({ m: final_scores_report.get(m, {}) for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in k_values]
    
    average_latency_ms = np.mean(latency_records) * 1000

    print("混合召回模型在【新格式专项测试集】上的评测结果:")
    print("-" * 70)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores_report.get('AUC', 0.0):.4f}")
    print(f"**平均查询处理时延 (分数融合+排序)**: {average_latency_ms:.4f} 毫秒/查询")
    print("-" * 70)
    
    # --- 7. 打印错误分析报告 ---
    print(f"\n\n--- 专项测试 Top-1 错误案例分析 (共 {len(error_cases)} 个错误) ---")
    NUM_ERROR_EXAMPLES_TO_PRINT = 10
    if not error_cases:
        print("🎉 恭喜！在专项测试集上没有发现 Top-1 错误案例！")
    else:
        for i, case in enumerate(error_cases[:NUM_ERROR_EXAMPLES_TO_PRINT]):
            print(f"\n--- 错误案例 {i+1}/{len(error_cases)} ---")
            print(f"  [查询 Query (plan_out)]: {case['Query']}")
            print(f"  [真实工具 Ground Truth]: {case['Ground Truth']}")
            print(f"  [预测工具 Prediction@1]: {case['Prediction@1']}")
            print(f"  [预测工具 Prediction@5]: {case['Prediction@5']}")
        if len(error_cases) > NUM_ERROR_EXAMPLES_TO_PRINT:
            print(f"\n... (仅显示前 {NUM_ERROR_EXAMPLES_TO_PRINT} 个错误案例) ...")
    print("-" * 70)

if __name__ == "__main__":
    main()