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

# --- 召回器和评测模块 (保持不变) ---
from rank_bm25 import BM25Okapi
import jieba
from sklearn.metrics import roc_auc_score


class ToolRetriever:
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
        
        # 获取所有分数的排序索引
        sorted_indices = np.argsort(all_scores)[::-1]
        
        # 根据排序索引获取Top-K的工具和它们的分数
        top_k_indices = sorted_indices[:top_k]
        retrieved_tools = [self.tool_definitions[i] for i in top_k_indices]
        retrieved_scores = all_scores[top_k_indices]

        return retrieved_tools, retrieved_scores, all_scores


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


# --- 语料库构建模块 (保持不变) ---
def get_synonyms():
    return {
        "购物车": ["采购车"],
        "收藏": ["喜欢", "想要", "关注", "收藏夹"],
        "查": ["找", "搜", "搜索", "查询"], 
        "找订单": ["查订单", "搜订单"],
        "看物流": ["查快递", "包裹进度"],
        "买的东西": ["我买的", "收到的", "购买记录"],
        "卖的东西": ["我卖的", "发出的", "售出记录"],
        "发票": ["开票", "报销凭证"],
    }

def build_corpus(data_df: pd.DataFrame, tool_definitions: list) -> list:
    print("--- 正在构建增强语料库... ---")
    synonyms = get_synonyms()
    
    tool_text_aggregator = defaultdict(list)
    for _, row in data_df.iterrows():
        if row['ground_truth_tool'] and isinstance(row['ground_truth_tool'], list):
            tool_name = row['ground_truth_tool'][0]['name']
            tool_text_aggregator[tool_name].append(row['指令'])
            
    all_tools_corpus = []
    for tool_def in tool_definitions:
        tool_name = tool_def['name']
        
        description = tool_def.get('description', '')
        aggregated_instructions = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
        
        synonym_expansion = []
        for word, syns in synonyms.items():
            if word in description:
                synonym_expansion.extend(syns)
        
        synonym_text = ' '.join(set(synonym_expansion))
        
        document = f"{aggregated_instructions}"
        all_tools_corpus.append(document)
        
    print(f"语料库构建完成，共 {len(all_tools_corpus)} 个工具。\n")
    return all_tools_corpus

# --- 主程序 (已修改) ---
def main():
    # --- 0. 配置区域 ---
    annotated_data_file_path = r'D:\Agent\code\bm25_recall\data\single_gt_output_with_plan.csv' 
    K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    NUM_ERROR_EXAMPLES_TO_PRINT = 10 
    # <--- 新增: 输出文件路径配置 ---
    OUTPUT_FILE_PATH = r'D:\Agent\code\bm25_recall\data\bm25_recall_results.csv'

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
    
    print(f"数据加载完成: 共 {len(data_df)} 条。所有数据将用于构建和评测。\n")


    # --- 2. 参数搜索 (Grid Search) ---
    print("--- 步骤 2: 在完整数据集上搜索最佳 BM25 参数 ---")
    all_tools_definitions = get_exact_tool_definitions()
    
    corpus = build_corpus(data_df, all_tools_definitions)
    
    best_score = -1
    best_params = {'k1': 1.5, 'b': 0.75}
    k1_range = [1.2, 1.5, 1.8, 2.0]
    b_range = [0.6, 0.75, 0.9]

    for k1, b in tqdm(list(itertools.product(k1_range, b_range)), desc="参数搜索中"):
        temp_retriever = ToolRetriever(corpus, all_tools_definitions, k1=k1, b=b)
        recalls_at_1 = []
        for _, row in data_df.iterrows():
            query = row['plan（在xx中做什么）']
            gt = row['ground_truth_tool']
            # <--- 修改: 适应新的 retrieve_with_scores 返回值 ---
            retrieved, _, _ = temp_retriever.retrieve_with_scores(query, top_k=1)
            recalls_at_1.append(calculate_recall_at_k(retrieved, gt, k=1))
        
        current_score = np.mean(recalls_at_1)
        if current_score > best_score:
            best_score = current_score
            best_params = {'k1': k1, 'b': b}

    print(f"参数搜索完成！最佳参数: {best_params} (在完整数据集上的 Recall@1: {best_score:.4f})\n")

    # --- 3. 构建最终召回器 ---
    print("--- 步骤 3: 使用最佳参数和完整数据构建最终召回器 ---")
    retriever = ToolRetriever(corpus, all_tools_definitions, k1=best_params['k1'], b=best_params['b'])
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
    # <--- 新增: 初始化用于存储详细召回结果的列表 ---
    detailed_predictions = []

    for i, (_, row) in enumerate(tqdm(data_df.iterrows(), total=len(data_df), desc="评测中")):
        query = row['plan（在xx中做什么）'] 
        original_query = row['query'] # 获取原始query
        ground_truth = row['ground_truth_tool']
        
        start_time = time.perf_counter()
        # <--- 修改: 接收新的返回值 ---
        retrieved_tools, retrieved_scores, all_scores = retriever.retrieve_with_scores(query, top_k=max(K_VALUES))
        duration = time.perf_counter() - start_time
        
        # <--- 新增: 收集详细的召回结果 ---
        prediction_record = {
            "query": original_query,
            "plan": query,
            "ground_truth": [_get_tool_names(ground_truth)],
            "retrieved_top_k": [{"tool": t.get('name'), "score": float(s)} for t, s in zip(retrieved_tools, retrieved_scores)]
        }
        detailed_predictions.append(prediction_record)

        results['processing_time'].append(duration)
        results['AUC'].append(calculate_auc_for_query(all_scores, all_tools_definitions, ground_truth))

        for k in K_VALUES:
            results['Recall@K'][k].append(calculate_recall_at_k(retrieved_tools, ground_truth, k))
            results['HR@K'][k].append(calculate_hit_ratio_at_k(retrieved_tools, ground_truth, k))
            results['MAP@K'][k].append(calculate_average_precision_at_k(retrieved_tools, ground_truth, k))
            results['MRR@K'][k].append(calculate_mrr_at_k(retrieved_tools, ground_truth, k))
            results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved_tools, ground_truth, k))
            results['COMP@K'][k].append(calculate_completeness_at_k(retrieved_tools, ground_truth, k))
        
        is_top1_correct = calculate_recall_at_k(retrieved_tools, ground_truth, k=1) >= 1.0
        if not is_top1_correct:
            gt_name = _get_tool_names(ground_truth).pop() if ground_truth else "N/A"
            pred_name_top1 = retrieved_tools[0].get('name') if retrieved_tools else "N/A"
            error_cases.append({
                "Query": query,
                "Ground Truth": [gt_name],
                "Prediction@1": [pred_name_top1],
                "Prediction@5": [r.get('name') for r in retrieved_tools[:5]]
            })

    # --- 5. 汇总并报告结果 ---
    print("\n\n--- 步骤 5: 评测结果报告 ---")
    final_scores = {}
    for metric, vals in results.items():
        if metric == 'AUC': final_scores['AUC'] = np.mean(vals)
        elif metric == 'processing_time': continue
        else: final_scores[metric] = {f"@{k}": np.mean(v) for k, v in vals.items()}
    report_df = pd.DataFrame({ m: final_scores[m] for m in ['Recall@K', 'HR@K', 'MAP@K', 'MRR@K', 'NDCG@K', 'COMP@K']}).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    print("BM25 召回模型在完整数据集上的评测结果:")
    print("-" * 70)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print(f"\n**AUC (全量排序 ROC AUC)**: {final_scores['AUC']:.4f}")
    print("-" * 70)
    
    total_time, avg_time_ms = np.sum(results['processing_time']), np.mean(results['processing_time']) * 1000
    qps = len(data_df) / total_time if total_time > 0 else 0
    print("\n性能评测:")
    print("-" * 70)
    print(f"样本总数: {len(data_df)} 条")
    print(f"总耗时: {total_time:.4f} 秒, 平均每条耗时: {avg_time_ms:.4f} 毫秒, QPS: {qps:.2f}")
    print("-" * 70)
    
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
    print("-" * 70)

    # <--- 新增: 步骤 7, 保存召回结果到文件 ---
    print(f"\n\n--- 步骤 7: 保存详细召回结果到文件 ---")
    try:
        output_records = []
        for pred in detailed_predictions:
            record = {
                'query': pred['query'],
                'plan': pred['plan'],
                'ground_truth': ', '.join(list(pred['ground_truth'][0])) if pred['ground_truth'] else '',
            }
            # 为Top-K的每个预测结果创建列
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
    main()