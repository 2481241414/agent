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
    # 不再需要 sentence_transformers 或 faiss
    import torch
    from rank_bm25 import BM25Okapi
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install torch transformers rank_bm25 scikit-learn pandas tqdm jieba")
    exit()

# ==============================================================================
# 区域 2: 召回器与精排器类定义
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

class QwenReranker:
    """【精排通路】使用 Qwen3-Reranker-0.6B 模型对召回结果进行重新排序。"""
    def __init__(self, model_name: str, instruction: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- [精排通路] 正在加载 Reranker 模型 (Causal LM): {model_name} ---")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()

        self.instruction = instruction
        self.max_length = 128
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        print(f"--- [精排通路] Reranker 模型加载完成，使用设备: {self.device} ---")

    @torch.no_grad()
    def compute_scores(self, query: str, documents: list) -> np.ndarray:
        if not documents:
            return np.array([])
        
        pairs_formatted = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in documents]

        inputs = self.tokenizer(
            pairs_formatted, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i in range(len(inputs['input_ids'])):
            inputs['input_ids'][i] = self.prefix_tokens + inputs['input_ids'][i] + self.suffix_tokens
        
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        batch_logits = self.model(**inputs).logits[:, -1, :]
        
        true_vector = batch_logits[:, self.token_true_id]
        false_vector = batch_logits[:, self.token_false_id]
        
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores_softmax = torch.nn.functional.log_softmax(batch_scores, dim=1)
        
        scores = batch_scores_softmax[:, 1].exp().cpu().numpy()
        
        return scores

# ==============================================================================
# 区域 3: 评测指标计算函数 (不变)
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
    return sum_prec / len(gt_names) if len(gt_names) > 0 else 0.0

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
    # ... 省略工具列表以减少篇幅 ...
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
# 区域 5: 评测核心逻辑 (已简化)
# ==============================================================================
def evaluate_with_reranker(
    data_df: pd.DataFrame, 
    bm25_retriever: BM25Retriever, 
    reranker: QwenReranker,
    all_tools_definitions: list,
    reranker_docs_by_index: list, 
    k_values: list,
    rerank_top_n: int, 
    full_report=False
):
    results = defaultdict(lambda: defaultdict(list))
    error_cases = []
    
    iterator = tqdm(data_df.iterrows(), total=len(data_df), desc="评测中(BM25召回+精排)")
    for i, row in iterator:
        query = row['plan（在xx中做什么）']
        ground_truth = row['ground_truth_tool']
        
        # --- 阶段 1: 仅使用 BM25 召回 ---
        initial_recall_scores = bm25_retriever.retrieve_scores(query)
        recall_indices = np.argsort(initial_recall_scores)[::-1]
        
        # --- 阶段 2: 精排 ---
        rerank_candidate_indices = recall_indices[:rerank_top_n]
        rerank_candidate_docs = [reranker_docs_by_index[i] for i in rerank_candidate_indices]
        
        reranker_scores = reranker.compute_scores(query, rerank_candidate_docs)
        
        # --- 阶段 3: 合并分数并最终排序 ---
        final_scores = np.full(len(all_tools_definitions), -float('inf'), dtype=np.float32)
        if len(reranker_scores) > 0:
            for idx, score in zip(rerank_candidate_indices, reranker_scores):
                final_scores[idx] = score
            
        sorted_indices = np.argsort(final_scores)[::-1]
        retrieved = [all_tools_definitions[idx] for idx in sorted_indices]
        
        # 计算指标
        if full_report:
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

    if full_report:
        return results, error_cases
    else: 
        # 此分支在当前主逻辑下不会被调用，但保留以备后用
        return np.mean(results['Recall@K'][1])

# ==============================================================================
# 区域 6: 主程序 (已简化)
# ==============================================================================
def main():
    # --- 0. 配置区域 ---
    annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_plan.csv'
    RERANKER_MODEL_NAME = '/home/workspace/lgq/shop/model/Qwen3-Reranker-0.6B'
    
    RERANKER_INSTRUCTION = 'Given a user query plan, determine if the following instruction text corresponds to the correct tool to execute that plan.'
    RERANK_TOP_N = 20
    K_VALUES = [1, 2, 3, 5, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10

    # --- 1. 数据加载 ---
    print("--- 步骤 1: 加载完整数据集 ---")
    try:
        required_columns = ['query', '指令', 'plan（在xx中做什么）', 'ground_truth_tool']
        data_df = pd.read_csv(annotated_data_file_path, usecols=required_columns).dropna().reset_index(drop=True)
        def parse_tools(s): return ast.literal_eval(s) if isinstance(s, str) else []
        data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(parse_tools)
        print(f"数据加载完成: 共 {len(data_df)} 条。\n")
    except FileNotFoundError:
        print(f"错误: 数据文件未找到，请检查路径 '{annotated_data_file_path}'")
        return
    except Exception as e:
        print(f"错误: 读取或解析文件 '{annotated_data_file_path}' 失败. {e}")
        return

    # --- 2. 初始化召回器和精排器 ---
    print("--- 步骤 2: 初始化 BM25 召回器和 Qwen Reranker 精排器 ---")
    all_tools_definitions = get_exact_tool_definitions()
    
    bm25_retriever = BM25Retriever(data_df, all_tools_definitions)
    reranker = QwenReranker(model_name=RERANKER_MODEL_NAME, instruction=RERANKER_INSTRUCTION)
    
    # --- 3. 为Reranker准备基于 '指令' 的文档 ---
    print("--- 步骤 3: 为Reranker准备基于 '指令' 的文档集 ---")
    tool_to_instructions_map = defaultdict(list)
    for _, row in data_df.iterrows():
        if isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
            tool_name = row['ground_truth_tool'][0]['name']
            if pd.notna(row['指令']):
                tool_to_instructions_map[tool_name].append(row['指令'])

    reranker_docs_by_index = []
    for tool in all_tools_definitions:
        tool_name = tool['name']
        instructions = tool_to_instructions_map.get(tool_name)
        if instructions:
            doc_text = ' '.join(set(instructions))
            reranker_docs_by_index.append(doc_text)
        else:
            reranker_docs_by_index.append(tool['description'])
    print("--- Reranker文档集准备完成 ---\n")
    
    # --- 4. 执行评测 ---
    print(f"\n--- 步骤 4: 使用 BM25 召回和 Rerank_Top_N={RERANK_TOP_N} 进行最终完整评测 ---")
    results, error_cases = evaluate_with_reranker(
        data_df, 
        bm25_retriever, 
        reranker,
        all_tools_definitions,
        reranker_docs_by_index,
        k_values=K_VALUES,
        rerank_top_n=RERANK_TOP_N,
        full_report=True
    )
    
    # --- 5. 汇总并报告最终结果 ---
    print("\n\n--- 步骤 5: 最终评测结果报告 ---")
    final_scores_report = {}
    for metric, vals in results.items():
        if metric == 'AUC': 
            final_scores_report['AUC'] = np.mean(vals['all'])
        else: 
            final_scores_report[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    
    report_df = pd.DataFrame({ m: final_scores_report[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    
    print(f"BM25召回 + Qwen Reranker (Top-N={RERANK_TOP_N}) 评测结果:")
    print("-" * 80)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores_report['AUC']:.4f}")
    print("-" * 80)
    
    # --- 6. 打印错误分析报告 ---
    print(f"\n\n--- 步骤 6: Top-1 错误案例分析 (共 {len(error_cases)} 个错误) ---")
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
    print("-" * 80)

if __name__ == "__main__":
    main()