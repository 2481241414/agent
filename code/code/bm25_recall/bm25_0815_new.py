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

# --- 召回器和评测模块 (评测模块不变, 召回器已修改) ---
from rank_bm25 import BM25Okapi
import jieba
from sklearn.metrics import roc_auc_score


# 【修改点】ToolRetriever 类被重构以适应新的逻辑
class ToolRetriever:
    """
    新的召回器逻辑：
    1. BM25 在一个由所有“指令”组成的语料库上构建索引。
    2. 当查询时，首先召回最相关的“指令”。
    3. 然后，将这些指令的分数聚合到它们各自对应的“工具”上。
    4. 最终返回得分最高的、去重后的工具列表。
    """
    def __init__(self, instruction_corpus: list, instruction_to_tool_map: list, all_tools_definitions: list, k1=1.5, b=0.75):
        self.instruction_corpus = instruction_corpus
        self.instruction_to_tool_map = instruction_to_tool_map # 记录每条指令对应的工具
        self.all_tools_definitions = all_tools_definitions
        self.k1 = k1
        self.b = b
        
        # 为了方便聚合分数，预先构建工具名到定义和索引的映射
        self.tool_name_to_def = {tool['name']: tool for tool in self.all_tools_definitions}
        self.tool_name_to_index = {tool['name']: i for i, tool in enumerate(self.all_tools_definitions)}
        
        # 预先计算每个工具对应哪些指令的索引，用于加速分数聚合
        self.tool_to_instruction_indices = defaultdict(list)
        for i, tool_list in enumerate(self.instruction_to_tool_map):
            if tool_list:
                # 假设每条指令只对应一个真实工具
                tool_name = tool_list[0]['name']
                self.tool_to_instruction_indices[tool_name].append(i)

        self._add_jieba_words()
        
        # 在指令语料库上构建 BM25 模型
        tokenized_corpus = [self._tokenize(doc) for doc in self.instruction_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def _tokenize(self, text: str) -> list[str]:
        return jieba.lcut(text, cut_all=False)

    def _add_jieba_words(self):
        # jieba 加载词典的逻辑保持不变
        for tool in self.all_tools_definitions:
            jieba.add_word(tool.get('name', ''), freq=100)
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "降价", "签到", "积分", "发票", "开票", "报销凭证"]
        for word in core_words:
            jieba.add_word(word, freq=100)

    def retrieve_with_scores(self, query: str, top_k: int):
        tokenized_query = self._tokenize(query)
        
        # 1. 计算查询与语料库中每条“指令”的相似度分数
        instruction_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. 聚合分数：将指令的分数赋给其对应的工具
        #    - 一个工具的分数，是所有指向它的指令分数中的最大值。
        #    - 这是为了解决 AUC 评测需要每个工具都有一个分数的问题。
        num_total_tools = len(self.all_tools_definitions)
        aggregated_tool_scores = np.zeros(num_total_tools)
        
        for tool_name, instruction_indices in self.tool_to_instruction_indices.items():
            if not instruction_indices:
                continue
            # 获取该工具对应的所有指令的分数，并取最大值
            max_score = np.max(instruction_scores[instruction_indices])
            tool_idx = self.tool_name_to_index[tool_name]
            aggregated_tool_scores[tool_idx] = max_score

        # 3. 根据聚合后的工具分数进行排序和筛选
        #    - `argsort` 返回的是从小到大的索引，所以需要倒序
        top_k_tool_indices = np.argsort(aggregated_tool_scores)[-top_k:][::-1]
        
        # 过滤掉分数为0的工具
        retrieved_tools = [
            self.all_tools_definitions[i] 
            for i in top_k_tool_indices 
            if aggregated_tool_scores[i] > 0
        ]
        
        return retrieved_tools, aggregated_tool_scores


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
    return sum_prec / len(gt_names)

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


# --- 工具定义模块 (保持不变) ---
def get_exact_tool_definitions():
    tools = [
        # ... (工具定义列表保持不变，此处省略以节省篇幅) ...
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


# --- 语料库构建模块 (已修改) ---
# 【修改点】重写语料库构建函数
def build_instruction_corpus_and_mapping(data_df: pd.DataFrame):
    """
    为数据集中的每一条指令创建一个文档。
    返回:
        - instruction_corpus (list[str]): 一个列表，每个元素是一条用户指令。
        - instruction_to_tool_map (list[list[dict]]): 一个列表，其索引与 corpus 对应，
          每个元素是该指令对应的 ground_truth_tool 列表。
    """
    print("--- 正在构建指令语料库和映射... ---")
    
    instruction_corpus = []
    instruction_to_tool_map = []

    for _, row in data_df.iterrows():
        # 使用 '指令' 列作为文档内容
        instruction_corpus.append(row['指令']) 
        instruction_to_tool_map.append(row['ground_truth_tool'])

    print(f"指令语料库构建完成，共 {len(instruction_corpus)} 条指令。\n")
    return instruction_corpus, instruction_to_tool_map


# --- 主程序 (已修改) ---
def main():
    # --- 0. 配置区域 ---
    annotated_data_file_path = r'D:\Agent\code\bm25_recall\data\single_gt_output_with_plan.csv' 
    K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10 

    # --- 1. 数据加载 ---
    print("--- 步骤 1: 加载完整数据集 ---")
    try:
        required_columns = ['query', '指令', 'plan（在xx中做什么）', 'ground_truth_tool']
        data_df = pd.read_csv(annotated_data_file_path, usecols=required_columns)
        data_df.dropna(subset=required_columns, inplace=True)
    except Exception as e:
        print(f"错误: 读取文件 '{annotated_data_file_path}' 失败. {e}")
        return
    
    def parse_tools(tool_string):
        try: return ast.literal_eval(tool_string)
        except (ValueError, SyntaxError): return []
    data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(parse_tools)
    # 过滤掉没有真实工具标注的数据
    data_df = data_df[data_df['ground_truth_tool'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    print(f"数据加载完成: 共 {len(data_df)} 条有效标注数据。所有数据将用于构建和评测。\n")


    # --- 2. 参数搜索 (Grid Search) ---
    print("--- 步骤 2: 在完整数据集上搜索最佳 BM25 参数 ---")
    all_tools_definitions = get_exact_tool_definitions()
    
    # 【修改点】使用新的函数构建语料库和映射
    instruction_corpus, instruction_to_tool_map = build_instruction_corpus_and_mapping(data_df)
    
    best_score = -1
    best_params = {'k1': 1.5, 'b': 0.75}
    k1_range = [1.2, 1.5, 1.8, 2.0]
    b_range = [0.6, 0.75, 0.9]

    for k1, b in tqdm(list(itertools.product(k1_range, b_range)), desc="参数搜索中"):
        # 【修改点】使用新的参数初始化临时召回器
        temp_retriever = ToolRetriever(instruction_corpus, instruction_to_tool_map, all_tools_definitions, k1=k1, b=b)
        recalls_at_1 = []
        
        for _, row in data_df.iterrows():
            # 【注意】我们使用 '指令' 列来评测召回效果，因为它与语料库的构建方式一致
            query = row['plan（在xx中做什么）']
            gt = row['ground_truth_tool']
            retrieved, _ = temp_retriever.retrieve_with_scores(query, top_k=1)
            recalls_at_1.append(calculate_recall_at_k(retrieved, gt, k=1))
        
        current_score = np.mean(recalls_at_1)
        if current_score > best_score:
            best_score = current_score
            best_params = {'k1': k1, 'b': b}

    print(f"参数搜索完成！最佳参数: {best_params} (在完整数据集上的 Recall@1: {best_score:.4f})\n")

    # --- 3. 构建最终召回器 ---
    print("--- 步骤 3: 使用最佳参数和完整数据构建最终召回器 ---")
    # 【修改点】使用新的方式和最佳参数构建最终召回器
    retriever = ToolRetriever(instruction_corpus, instruction_to_tool_map, all_tools_definitions, k1=best_params['k1'], b=best_params['b'])
    print("最终召回器构建完成。\n")

    # --- 4. 在完整数据集上评测 ---
    print(f"--- 步骤 4: 开始在 {len(data_df)} 条样本上进行评测 ---")
    results = {
        'Recall@K': {k: [] for k in K_VALUES}, 'HR@K': {k: [] for k in K_VALUES},
        'MAP@K': {k: [] for k in K_VALUES}, 'MRR@K': {k: [] for k in K_VALUES},
        'NDCG@K': {k: [] for k in K_VALUES}, 'COMP@K': {k: [] for k in K_VALUES},
        'AUC': [], 'processing_time': []
    }
    error_cases = []

    for i, (_, row) in enumerate(tqdm(data_df.iterrows(), total=len(data_df), desc="评测中")):
        # 【注意】评测时， query 可以是 'plan' 或 '指令'，取决于你想模拟的真实场景
        # 这里我们继续使用 '指令' 作为查询，以保持和参数搜索时的一致性
        query = row['plan（在xx中做什么）'] 
        ground_truth = row['ground_truth_tool']
        
        start_time = time.perf_counter()
        retrieved, all_scores = retriever.retrieve_with_scores(query, top_k=max(K_VALUES))
        duration = time.perf_counter() - start_time
        
        results['processing_time'].append(duration)
        # AUC评测函数现在可以正确工作，因为它接收的是聚合后的、针对每个工具的分数
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
                "Prediction@5": [r.get('name') for r in retrieved[:5]]
            })

    # --- 5. 汇总并报告结果 ---
    print("\n\n--- 步骤 5: 评测结果报告 (新逻辑) ---")
    final_scores = {}
    for metric, vals in results.items():
        if metric == 'AUC': final_scores['AUC'] = np.mean(vals)
        elif metric == 'processing_time': continue
        else: final_scores[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    report_df = pd.DataFrame({ m: final_scores[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    print("BM25 召回模型 (基于指令搜索) 在完整数据集上的评测结果:")
    print("-" * 50)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores['AUC']:.4f}")
    print("-" * 50)
    
    total_time, avg_time_ms = np.sum(results['processing_time']), np.mean(results['processing_time']) * 1000
    qps = len(data_df) / total_time if total_time > 0 else 0
    print("\n性能评测:")
    print("-" * 50)
    print(f"样本总数: {len(data_df)} 条")
    print(f"总耗时: {total_time:.4f} 秒, 平均每条耗时: {avg_time_ms:.4f} 毫秒, QPS: {qps:.2f}")
    print("-" * 50)
    
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
    print("-" * 50)


if __name__ == "__main__":
    main()