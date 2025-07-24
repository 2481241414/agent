import pandas as pd
import os
import json
import ast
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict

# --- 召回器和评测模块 (这部分代码保持不变) ---
from rank_bm25 import BM25Okapi
import jieba

class ToolRetriever:
    def __init__(self, all_tools_corpus: list, all_tools_definitions: list):
        self.tool_corpus = all_tools_corpus
        self.tool_definitions = all_tools_definitions
        tokenized_corpus = [self._tokenize(doc) for doc in self.tool_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        return jieba.lcut(text)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = doc_scores.argsort()[-top_k:][::-1]
        return [self.tool_definitions[i] for i in top_k_indices if doc_scores[i] > 0]

def _get_tool_names(tools: list) -> set:
    if not isinstance(tools, list): return set()
    return {tool.get('name') for tool in tools}

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    if not ground_truth: return 1.0
    retrieved_names_at_k = _get_tool_names(retrieved[:k])
    ground_truth_names = _get_tool_names(ground_truth)
    if not ground_truth_names: return 1.0 # Avoid division by zero
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

# --- 主程序 ---
def main():
    # --- 0. 配置区域 ---
    data_file_path = r'D:\Agent\data\generated_queries - 0704手动筛选后.csv'
    mapping_file_path = r'D:\Agent\data\大类-工具映射关系表-0707-Cleaned.csv'
    K_VALUES = [1, 3, 5]
    
    # ★★★ 新增配置：控制打印的样本数量 ★★★
    NUM_EXAMPLES_TO_PRINT = 1  # 打印前10个样本的详细情况，设为0则不打印

    # --- 1. 数据准备 --- (这部分不变)
    print("--- 步骤 1: 准备评测数据 ---")
    try:
        required_columns = ['category', 'app_name', 'instruction_template', 'final_query', 'is_train']
        data_df = pd.read_csv(data_file_path, usecols=required_columns)
        mapping_df = pd.read_csv(mapping_file_path)
    except Exception as e:
        print(f"错误: 读取文件失败. {e}")
        return

    # --- 2. 构建丰富的工具描述 --- (这部分不变)
    print("--- 步骤 2: 构建 BM25 语料库 ---")
    def get_tool_names_for_row(row, mapping_df):
        filtered_df = mapping_df[
            (mapping_df['app'] == row['app_name']) &
            (mapping_df['大类'] == row['category'])
        ]
        return filtered_df['function_name'].unique().tolist()
    
    data_df['temp_tool_names'] = data_df.apply(get_tool_names_for_row, axis=1, args=(mapping_df,))

    tool_text_aggregator = defaultdict(list)
    for _, row in data_df.iterrows():
        for tool_name in row['temp_tool_names']:
            tool_text_aggregator[tool_name].append(row['instruction_template'])
            # tool_text_aggregator[tool_name].append(row['final_query'])
            # tool_text_aggregator[tool_name].append(row['app_name'])

    all_tools_corpus = []
    all_tools_definitions = []
    unique_tool_names = mapping_df['function_name'].dropna().unique()

    for name in unique_tool_names:
        description = ' '.join(set(tool_text_aggregator.get(name, [])))
        document = f"{name} {description}"
        all_tools_corpus.append(document)
        all_tools_definitions.append({'name': name, 'description': description})
    print(all_tools_definitions[:2])
    print(f"语料库构建完成，共 {len(all_tools_corpus)} 个工具。\n")

    # --- 3. 构建召回器 --- (这部分不变)
    print("--- 步骤 3: 构建 BM25 召回器 ---")
    retriever = ToolRetriever(all_tools_corpus, all_tools_definitions)
    print("召回器构建完成。\n")

    # --- 4. 准备 Ground Truth 并评测 (★★ 此处有修改 ★★) ---
    print(f"--- 步骤 4: 开始在 {len(data_df)} 个样本上进行评测 (K={K_VALUES}) ---")
    data_df['available_tools'] = data_df['temp_tool_names'].apply( 
        lambda names: [{'name': name} for name in names]
    )
    eval_df = data_df[data_df['available_tools'].apply(len) > 0].copy()

    results = {metric: {k: [] for k in K_VALUES} for metric in ['Recall@K', 'NDCG@K', 'COMP@K']}
    
    # 使用 enumerate 获取循环索引 i
    for i, (_, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc="评测中")):
        query = row['final_query']
        ground_truth = row['available_tools']
        
        # 召回数量为K值中的最大值，以确保所有K值的评测都有效
        max_k = max(K_VALUES)
        retrieved_tools = retriever.retrieve(query, top_k=max_k)
        
        if -1 < i < NUM_EXAMPLES_TO_PRINT:
            # 为了不打乱tqdm的进度条，先加两个换行符
            print(f"\n\n==================== 评测样本 {i+1} ====================")
            print(f" [查询 Query]: {query}")
            
            # 格式化输出，只显示工具名称列表，更清晰
            gt_names = [t.get('name', 'N/A') for t in ground_truth]
            print(f" [真实工具 Ground Truth]: {gt_names}")
            
            retrieved_names = [t.get('name', 'N/A') for t in retrieved_tools]
            print(f" [召回结果 Prediction @{max_k}]: {retrieved_names}")
            
            # 简单判断本次召回是否成功（以COMP@max_k为标准）
            is_complete = calculate_completeness_at_k(retrieved_tools, ground_truth, k=max_k)
            status = "✅ 完整命中 (Complete Hit)" if is_complete else "❌ 部分或偏离 (Partial/Miss)"
            print(f" [评测状态 @{max_k}]: {status}")
            print("=====================================================")

        # 计算指标的逻辑不变
        for k in K_VALUES:
            results['Recall@K'][k].append(calculate_recall_at_k(retrieved_tools, ground_truth, k))
            results['NDCG@K'][k].append(calculate_ndcg_at_k(retrieved_tools, ground_truth, k))
            results['COMP@K'][k].append(calculate_completeness_at_k(retrieved_tools, ground_truth, k))
            
    # --- 5. 汇总并报告结果 ---
    print("\n\n--- 步骤 5: 评测结果报告 ---")
    final_scores = {metric: {k: np.mean(scores) for k, scores in k_scores.items()} for metric, k_scores in results.items()}
    report_df = pd.DataFrame(final_scores).T
    report_df.columns = [f"@{k}" for k in K_VALUES]
    
    print("BM25 召回模型评测结果:")
    print("-" * 40)
    print(report_df.to_string(formatters={col: '{:.4f}'.format for col in report_df.columns}))
    print("-" * 40)

if __name__ == "__main__":
    main()