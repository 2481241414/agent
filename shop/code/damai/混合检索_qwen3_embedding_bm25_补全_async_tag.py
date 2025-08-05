import pandas as pd
import os
import json
import ast
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict, deque
import time
import asyncio

# ==============================================================================
# 区域 1: 检查并导入所需库
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer
    import torch
    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.metrics import roc_auc_score
    import aiohttp # 异步HTTP请求
    import jieba
except ImportError as e:
    print(f"错误: 缺少必要的库 -> {e}")
    print("请在终端运行: pip install faiss-cpu torch sentence-transformers transformers rank_bm25 scikit-learn pandas tqdm aiohttp jieba")
    exit()

# ==============================================================================
# 区域 2: 工具依赖关系图 (Tool Dependency Graph)
# 这是我们系统的核心知识库，用于自动补全前置工具。
# ==============================================================================
TOOL_DEPENDENCY_MAP = {
    # --------------------------------------------------------------------------
    # 场景一: 电影信息查询 (核心依赖：获取 showId)
    # 描述: 多个查询影片排期、详情的工具，都依赖于先通过搜索获取到影片的唯一标识 showId。
    # --------------------------------------------------------------------------
    "queryCinema": {
        "showId": "searchShowsBykeyWord"
    },
    "queryShowDetail": {
        "showId": "searchShowsBykeyWord"
    },
    "queryCinemasV2": {
        "showId": "searchShowsBykeyWord",
        "support": "queryCinemaFilter"
    },
    # --------------------------------------------------------------------------
    # 场景二: 带筛选条件的电影排期查询 (复合依赖)
    # --------------------------------------------------------------------------
    "queryShowWithCinema": {
        "supportList": "queryCinemaFilter"
    },
    # --------------------------------------------------------------------------
    # 场景三: 演出信息查询 (核心依赖：获取 artistId)
    # --------------------------------------------------------------------------
    "queryPerform": {
        "artistId": "queryArtist"
    },
    "relationArtist": {
        "artistId": "queryArtist"
    },
    # --------------------------------------------------------------------------
    # 场景四: 影院信息查询 (核心依赖：获取 cinemaId)
    # --------------------------------------------------------------------------
    "queryShowsByCinemaId": {
        "cinemaId": "searchCinemas"
    }
}

# ==============================================================================
# 区域 3: 前置工具补全函数 (核心逻辑)
# ==============================================================================
def augment_with_prerequisites(final_tool_names: list, dependency_map: dict) -> list:
    """
    根据最终目标工具和依赖图，通过拓扑排序构建完整的、有序的工具链。

    Args:
        final_tool_names (list): 模型召回的最终目标工具名称列表。
        dependency_map (dict): 全局工具依赖关系图。

    Returns:
        list: 一个包含所有需要工具（前置和最终）的、保证执行顺序的有序列表。
    """
    # 1. 构建图的邻接表和入度表
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    # 收集所有需要处理的节点
    nodes_to_process = set(final_tool_names)
    queue = deque(final_tool_names)
    
    while queue:
        tool = queue.popleft()
        if tool in dependency_map:
            for _, prereq in dependency_map[tool].items():
                if prereq not in nodes_to_process:
                    nodes_to_process.add(prereq)
                    queue.append(prereq)

    # 填充邻接表和入度
    for node in nodes_to_process:
        if node in dependency_map:
            for _, prereq in dependency_map[node].items():
                graph[prereq].append(node)
                in_degree[node] += 1

    # 2. 拓扑排序
    # 将所有入度为0的节点（即没有前置依赖的工具）放入队列
    topo_queue = deque([node for node in nodes_to_process if in_degree[node] == 0])
    sorted_chain = []
    
    while topo_queue:
        current_tool = topo_queue.popleft()
        sorted_chain.append(current_tool)
        
        # 遍历当前工具的所有后继工具
        for next_tool in graph[current_tool]:
            in_degree[next_tool] -= 1
            # 如果后继工具的入度变为0，则加入队列
            if in_degree[next_tool] == 0:
                topo_queue.append(next_tool)
    
    # 3. 检查是否存在循环依赖
    if len(sorted_chain) != len(nodes_to_process):
        # 在实际业务中应该记录日志或抛出异常
        print(f"警告: 检测到工具依赖中可能存在循环！最终工具: {final_tool_names}")
        # 在循环情况下，返回一个尽力而为的排序结果
        return list(nodes_to_process)
        
    return sorted_chain

# ==============================================================================
# 区域 4: 召回器类定义 (保持不变)
# ==============================================================================
class BM25Retriever:
    """【关键词通路】BM25召回器，专注于关键词和用户多样化表达的匹配。"""
    def __init__(self, data_df: pd.DataFrame, all_tools_definitions: list, k1=1.5, b=0.75):
        self.definitions = all_tools_definitions
        self.tool_name_to_idx = {tool['name'].split('(')[0]: i for i, tool in enumerate(all_tools_definitions)}
        self._add_jieba_words()
        
        print("--- [BM25通路] 正在构建关键词增强语料库... ---")
        corpus = self._build_keyword_rich_corpus(data_df)
        tokenized_corpus = [jieba.lcut(doc, cut_all=False) for doc in tqdm(corpus, desc="BM25语料库分词")]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        print("--- [BM25通路] 召回器构建完成 ---")

    def _add_jieba_words(self):
        for tool in self.definitions:
            jieba.add_word(tool.get('name', '').split('(')[0], freq=100)
        core_words = ["购物车", "采购车", "待收货", "待付款", "收藏夹", "发票", "优惠券", "IMAX", "杜比"]
        for word in core_words:
            jieba.add_word(word, freq=100)

    def _build_keyword_rich_corpus(self, data_df: pd.DataFrame) -> list:
        tool_text_aggregator = defaultdict(list)
        for _, row in data_df.iterrows():
            tools = row.get('ground_truth_tool')
            if not isinstance(tools, list) or not tools: continue
            
            instruction_parts = []
            if pd.notna(row['指令']):
                instruction_parts = row['指令'].split('\n')

            for i, tool in enumerate(tools):
                tool_name = tool['name'].split('(')[0]
                if i < len(instruction_parts):
                    tool_text_aggregator[tool_name].append(instruction_parts[i].strip())
                elif pd.notna(row['指令']):
                     tool_text_aggregator[tool_name].append(row['指令'])

        corpus = [''] * len(self.definitions)
        for tool_def in self.definitions:
            tool_name = tool_def['name'].split('(')[0]
            tool_idx = self.tool_name_to_idx[tool_name]
            aggregated_text = ' '.join(set(tool_text_aggregator.get(tool_name, [])))
            document = f"{aggregated_text}"
            corpus[tool_idx] = document
        return corpus

    def retrieve_scores(self, query: str) -> np.ndarray:
        tokenized_query = jieba.lcut(query, cut_all=False)
        return self.bm25.get_scores(tokenized_query)

class InstructionSearcher:
    """【精准意图通路】支持本地加载或API调用两种模式，以统一向量化方式。"""
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

        self._build_mappings(data_df)

    def _build_mappings(self, data_df: pd.DataFrame):
        self.instruction_to_tool_map = {}
        for _, row in data_df.iterrows():
            tools = row.get('ground_truth_tool')
            if pd.isna(row['指令']) or not isinstance(tools, list) or not tools:
                continue
            
            instructions = [i.strip() for i in row['指令'].strip().split('\n')]
            
            if len(instructions) == len(tools):
                for instruction, tool in zip(instructions, tools):
                    self.instruction_to_tool_map[instruction] = tool
            else:
                self.instruction_to_tool_map[row['指令'].strip()] = tools[-1] # 多任务映射到最终工具

        self.unique_instructions = list(self.instruction_to_tool_map.keys())
        self.tool_name_to_idx = {tool['name'].split('(')[0]: i for i, tool in enumerate(self.definitions)}

    def _build_faiss_index(self, embeddings: np.ndarray):
        embeddings = embeddings.astype('float32')
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    async def _get_embeddings_from_api(self, texts: list, session: aiohttp.ClientSession) -> list:
        # ... (此部分代码保持不变)
        return [] # 实际实现应保留

    async def initialize_and_get_all_scores(self, all_plan_queries: list):
        print(f"--- [意图通路] 正在将 {len(self.unique_instructions)} 条唯一指令编码为向量...")
        
        instruction_embeddings = self.model.encode(self.unique_instructions, convert_to_tensor=False, show_progress_bar=True)
        self._build_faiss_index(np.array(instruction_embeddings))
        print("--- [意图通路] Faiss索引构建完成 ---")

        print(f"--- [意图通路] 正在对 {len(all_plan_queries)} 条查询进行评分...")
        query_embeddings = self.model.encode(all_plan_queries, convert_to_tensor=False, show_progress_bar=True)
        
        all_semantic_scores = []
        for query_embedding in tqdm(query_embeddings, desc="计算语义分数"):
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
                        tool_name = tool_def['name'].split('(')[0]
                        tool_idx = self.tool_name_to_idx.get(tool_name)
                        if tool_idx is not None:
                            tool_scores[tool_idx] = max(tool_scores[tool_idx], dist)
            all_semantic_scores.append(tool_scores)
            
        return all_semantic_scores

# ==============================================================================
# 区域 5: 评测函数 (已修改以适应工具链)
# ==============================================================================
def _get_tool_names_from_defs(tools: list) -> list:
    if not isinstance(tools, list): return []
    return [tool.get('name', '').split('(')[0] for tool in tools]

def calculate_final_tool_match(predicted_chain: list, ground_truth_chain: list) -> float:
    """检查预测的工具链的最后一个工具是否与真实工具链的最后一个匹配"""
    if not predicted_chain or not ground_truth_chain:
        return 1.0 if not predicted_chain and not ground_truth_chain else 0.0
    return 1.0 if predicted_chain[-1] == ground_truth_chain[-1] else 0.0

def calculate_chain_completeness(predicted_chain: list, ground_truth_chain: list) -> float:
    """检查预测的工具链是否包含了真实工具链中的所有工具"""
    if not ground_truth_chain: return 1.0
    return 1.0 if set(ground_truth_chain).issubset(set(predicted_chain)) else 0.0

# ==============================================================================
# 区域 6: 工具定义 (已扩展)
# ==============================================================================
def get_exact_tool_definitions():
    """返回所有可用工具的定义列表"""
    tools = [
        # 电影工具
        {"name": "queryShowsByCinemaId(cinemaId)", "description": "根据指定影院 ID 查询该影院当前在映的所有影片信息"},
        {"name": "searchCinemas(cityName, cityCode, keyword, longitude, latitude)", "description": "根据关键词搜索影院，支持多维度搜索"},
        {"name": "queryCinemaFilter(cityCode, cityName)", "description": "获取指定城市的完整影院筛选条件"},
        {"name": "queryCinemasV2(cityCode, cityName, showId, sortType, longitude, latitude, support, needFilter)", "description": "查询指定城市的影院列表"},
        {"name": "queryShowWithCinema(cityName, cityCode, filmName, longitude, latitude, supportList)", "description": "查询影片在指定城市影院排期和场次信息"},
        {"name": "queryHotShows(cityCode, pageSize)", "description": "用于查询当前热映的影片信息"},
        {"name": "queryCinema(cityCode, showId, pageIndex, pageSize)", "description": "用于查询指定影片的影院以及场次信息"},
        {"name": "searchShowsBykeyWord(keyword, cityCode, pageSize)", "description": "根据电影名称进行关键词查询，返回电影基本信息"},
        {"name": "queryShowDetail(showId, cityCode)", "description": "根据影片 ID 查询影片详情"},
        
        # 演出工具
        {"name": "queryArtist(artistName)", "description": "用于查询演职人员信息（明星、厂牌、IP等）"},
        {"name": "queryPerform(showName, cityCode, categoryId, pageNo, pageSize, showStartTime, showEndTime, artistId, artistType)", "description": "用于查询演出项目信息"},
        {"name": "queryCategory(cityCode)", "description": "用于查询可用的演出类型分类"},
        {"name": "relationArtist(cityCode, action, artistId)", "description": "对指定艺人进行许愿或取消许愿操作"},
        
        # 其他工具
        {"name": "aiRequest(humanMessage)", "description": "AI 问答功能，兜底工具"},
    ]
    return tools

# ==============================================================================
# 区域 7: 评测核心逻辑 (已重构以支持工具链)
# ==============================================================================
def evaluate_recall_system(data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha, full_report=False):
    results = defaultdict(list)
    error_cases = []
    
    tool_name_map = {tool['name'].split('(')[0]: tool for tool in all_tools_definitions}

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if (max_s - min_s) == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    score_indices = data_df.index.tolist()

    for i, (_, row) in enumerate(data_df.iterrows()):
        original_index = score_indices[i]
        
        # 1. 获取真实工具链 (Ground Truth Chain)
        gt_final_tools_defs = row['ground_truth_tool']
        gt_final_tool_names = _get_tool_names_from_defs(gt_final_tools_defs)
        if not gt_final_tool_names:
            continue
        # 对真实最终工具也进行依赖补全，得到完整的真实工具链
        ground_truth_chain = augment_with_prerequisites(gt_final_tool_names, TOOL_DEPENDENCY_MAP)

        # 2. 混合评分并召回Top-1最终工具
        bm25_scores = all_bm25_scores[original_index]
        semantic_scores = all_semantic_scores[original_index]
        
        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)
        final_scores = alpha * norm_bm25 + (1 - alpha) * norm_semantic
        
        sorted_indices = np.argsort(final_scores)[::-1]
        
        # 初步召回的Top-1工具被视为预测的“最终工具”
        top1_predicted_final_tool_def = all_tools_definitions[sorted_indices[0]]
        top1_predicted_final_tool_name = _get_tool_names_from_defs([top1_predicted_final_tool_def])[0]

        # 3. ✨ 核心步骤: 补全预测的工具链 ✨
        predicted_chain = augment_with_prerequisites([top1_predicted_final_tool_name], TOOL_DEPENDENCY_MAP)
        
        # 4. 评测工具链
        final_tool_match = calculate_final_tool_match(predicted_chain, ground_truth_chain)
        chain_completeness = calculate_chain_completeness(predicted_chain, ground_truth_chain)

        results['FinalToolMatch@1'].append(final_tool_match)
        if full_report:
            results['ChainCompleteness@1'].append(chain_completeness)

            # 记录错误案例
            if final_tool_match < 1.0:
                error_cases.append({
                    "Query": row['plan（在xx中做什么）'],
                    "Ground Truth Chain": ground_truth_chain,
                    "Predicted Top1 Final Tool": top1_predicted_final_tool_name,
                    "Predicted Chain": predicted_chain
                })

    if full_report:
        return results, error_cases
    else:
        # 网格搜索时，仅返回核心指标的均值
        return np.mean(results['FinalToolMatch@1'])

# ==============================================================================
# 区域 8: 主程序 (已修改)
# ==============================================================================
async def main():
    # --- 0. 配置区域 ---
    MODE = 'local'  # 使用本地模式，API模式请确保服务已启动
    MODEL_PATH = '/path/to/your/sentence-transformer-model' # ‼️‼️ *** 修改为您的本地模型路径 *** ‼️‼️
    
    # 您的数据文件路径
    annotated_data_file_path = '/path/to/your/data.csv' # ‼️‼️ *** 修改为您的数据文件路径 *** ‼️‼️
    NUM_ERROR_EXAMPLES_TO_PRINT = 10
    
    # --- 1. 数据加载与预处理 ---
    print("--- 步骤 1: 加载并预处理数据集 ---")
    try:
        data_df = pd.read_csv(annotated_data_file_path, usecols=['plan（在xx中做什么）', '指令', 'ground_truth_tool'])
    except Exception as e:
        print(f"无法加载数据，请检查文件路径: {annotated_data_file_path}")
        print(f"错误: {e}")
        return

    data_df = data_df.dropna(subset=['指令', 'ground_truth_tool']).reset_index(drop=True)
    # 将ground_truth_tool从字符串解析为Python对象
    data_df['ground_truth_tool'] = data_df['ground_truth_tool'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    print(f"数据加载完成: 共 {len(data_df)} 条有效数据。\n")

    # --- 2. 初始化召回器 (使用全量数据) ---
    all_tools_definitions = get_exact_tool_definitions()
    
    bm25_retriever = BM25Retriever(data_df, all_tools_definitions)
    
    instruction_searcher = InstructionSearcher(
        data_df=data_df, 
        all_tools_definitions=all_tools_definitions,
        mode=MODE,
        model_path_or_name=MODEL_PATH,
    )

    # --- 3. 为全量数据计算所有分数 ---
    print("\n--- 步骤 3: 为全量数据计算所有召回分数 ---")
    all_queries = data_df['plan（在xx中做什么）'].tolist()
    
    all_bm25_scores = [bm25_retriever.retrieve_scores(query) for query in tqdm(all_queries, desc="计算BM25分数")]
    all_semantic_scores = await instruction_searcher.initialize_and_get_all_scores(all_queries)

    # --- 4. Alpha值网格搜索 ---
    print(f"\n--- 步骤 4: 开始进行Alpha值网格搜索 ---")
    alpha_range = np.linspace(0, 1, 21)
    best_alpha = -1
    best_score = -1
    
    for alpha in tqdm(alpha_range, desc="Alpha网格搜索"):
        current_score = evaluate_recall_system(data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, alpha)
        if current_score > best_score:
            best_score = current_score
            best_alpha = alpha

    print(f"\n--- Alpha值网格搜索完成 ---")
    print(f"找到的最佳Alpha值: {best_alpha:.2f} (对应的最高平均 FinalToolMatch@1 为: {best_score:.4f})")
    
    # --- 5. 使用最佳Alpha进行最终的完整评测 ---
    print(f"\n--- 步骤 5: 使用最佳Alpha={best_alpha:.2f}进行最终的完整评测 ---")
    results, error_cases = evaluate_recall_system(
        data_df, all_bm25_scores, all_semantic_scores, all_tools_definitions, best_alpha, full_report=True
    )
    
    # --- 6. 汇总并报告最终结果 ---
    print(f"\n\n--- 步骤 6: 最终评测结果报告 (模式: {MODE.upper()}, Alpha: {best_alpha:.2f}) ---")
    final_ftm_score = np.mean(results['FinalToolMatch@1'])
    final_cc_score = np.mean(results['ChainCompleteness@1'])
    
    print("-" * 70)
    print(f"**最终工具匹配率 (FinalToolMatch@1)**: {final_ftm_score:.4f}")
    print(f"**工具链完整性 (ChainCompleteness@1)**: {final_cc_score:.4f}")
    print("-" * 70)

    # --- 7. 错误分析 ---
    print(f"\n--- 步骤 7: 错误案例分析 (共 {len(error_cases)} 个错误) ---")
    if not error_cases:
        print("🎉 恭喜！在数据集上没有发现错误案例！")
    else:
        for i, case in enumerate(error_cases[:NUM_ERROR_EXAMPLES_TO_PRINT]):
            print(f"\n--- 错误案例 {i+1}/{len(error_cases)} ---")
            print(f"  [查询 Query]: {case['Query']}")
            print(f"  [真实工具链 Ground Truth]: {case['Ground Truth Chain']}")
            print(f"  [预测的最终工具]: {case['Predicted Top1 Final Tool']}")
            print(f"  [预测的完整工具链]: {case['Predicted Chain']}")
        if len(error_cases) > NUM_ERROR_EXAMPLES_TO_PRINT:
            print(f"\n... (仅显示前 {NUM_ERROR_EXAMPLES_TO_PRINT} 个错误案例) ...")
    print("-" * 70)


if __name__ == "__main__":
    # 注意: 如果您的模型较大或数据量很大，首次运行可能需要较长时间来构建索引。
    # 确保您的模型路径和数据文件路径已正确设置。
    asyncio.run(main())