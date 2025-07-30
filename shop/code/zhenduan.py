import pandas as pd
import ast
import os

# ==============================================================================
# 目标：找出在主代码处理流程中，究竟是哪条指令被“丢失”了
# ==============================================================================

def diagnose_missing_instruction():
    """
    通过模拟主代码的数据处理流程，精确对比前后差异，找出丢失的指令。
    """
    
    # --- 1. 配置您的数据文件路径 ---
    # 这个是你主代码中引用的、已经生成好的CSV文件
    # annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_plan_0815_完整.csv'
    annotated_data_file_path = '/home/workspace/lgq/shop/data/single_gt_output_with_fc_0815_能力.csv'

    if not os.path.exists(annotated_data_file_path):
        print(f"错误：诊断文件未找到 -> {annotated_data_file_path}")
        return

    print("--- 开始执行深度诊断 ---")

    # --- 2. 步骤一: 获取处理前的“指令”全集 ---
    # 我们直接从CSV读取，不做任何处理，获取最原始的、应该有的1034条唯一指令。
    df_raw = pd.read_csv(annotated_data_file_path)
    instructions_set_before = set(df_raw['指令'].dropna().unique())
    print(f"\n[诊断步骤1] 在任何处理之前，文件中有 {len(instructions_set_before)} 条唯一指令。")

    # --- 3. 步骤二: 完整模拟您的主代码处理流程 ---
    # 我们将完全复现主代码中从加载到构建 unique_instructions 的每一步。
    
    # (a) 从主代码中复制过来的解析函数
    def parse_tools(s):
        try:
            # 必须用 try-except 包裹，因为 ast.literal_eval 对格式非常敏感
            return ast.literal_eval(s) if isinstance(s, str) else []
        except (ValueError, SyntaxError):
            # 如果字符串不是一个有效的Python字面量（比如格式错误的JSON），就返回空列表
            return []

    # (b) 加载数据并应用解析函数，就像主代码一样
    df_processed = pd.read_csv(annotated_data_file_path)
    df_processed['ground_truth_tool'] = df_processed['ground_truth_tool'].apply(parse_tools)

    # (c) 完全复制 InstructionSearcher._build_mappings 的核心逻辑
    instructions_set_after = set()
    # 我们使用 'first'，因为之前的尝试已经排除了 'last'
    for _, row in df_processed.drop_duplicates(subset=['指令'], keep='first').iterrows():
        # 这个 if 条件就是您的代码中过滤指令的核心
        if pd.notna(row['指令']) and isinstance(row.get('ground_truth_tool'), list) and row['ground_truth_tool']:
            instructions_set_after.add(row['指令'])

    print(f"[诊断步骤2] 模拟主代码的处理流程后，剩下 {len(instructions_set_after)} 条唯一指令。")

    # --- 4. 步骤三: 计算差集，锁定元凶 ---
    print("\n--- 诊断结果 ---")
    missing_instructions = instructions_set_before - instructions_set_after

    if not missing_instructions:
        print("✅ 诊断未发现差异。这非常奇怪，请检查文件路径是否完全一致。")
    else:
        print(f"🔥 找到了！您的代码流程丢失了 {len(missing_instructions)} 条指令。")
        
        for i, missing_instr in enumerate(missing_instructions):
            print(f"\n--- 丢失的指令 #{i+1} ---")
            print(f"指令内容: '{missing_instr}'")
            
            # 在最原始的DataFrame中找到这一行，看看它到底长什么样
            offending_row = df_raw[df_raw['指令'] == missing_instr]
            
            print("\n它在您的CSV文件中的原始数据是:")
            print(offending_row.to_string())
            
            print("\n[问题分析]:")
            print("请仔细查看上面这行数据的 'ground_truth_tool' 列。问题很可能出在这里：")
            print("1. 这个单元格的值可能不是一个有效的列表字符串，导致 `parse_tools` 函数解析失败，返回了 `[]` (空列表)。")
            print("2. `_build_mappings` 函数中的 `if ... and row['ground_truth_tool']` 条件判断，对于空列表 `[]` 会判定为 False。")
            print("结论: 这条指令因为其关联的 'ground_truth_tool' 数据有问题，所以在构建映射表时被过滤掉了。")

# --- 运行诊断程序 ---
if __name__ == "__main__":
    diagnose_missing_instruction()