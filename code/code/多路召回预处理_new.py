import pandas as pd
import os
import json
import csv
import sys

# ==============================================================================
# 区域 1: 全局工具定义 (来自原脚本1)
# ==============================================================================
def get_exact_tool_definitions():
    """
    定义了系统中所有可用的工具及其完整的函数签名和描述。
    这是所有工具的“真实来源”(Source of Truth)。
    """
    tools = [
        {"name": "open_orders_bought(app, order_status)", "description": "在app应用程序中查看买入的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_sold(app, order_status)", "description": "在app应用程序中查看自己售卖的指定状态的订单列表，例如待付款、待收货、待评价等。"},
        {"name": "open_orders_all_review(app)", "description": "在app应用程序中查看待评价状态的订单列表，在不指定购买还是售卖的订单时，及全都要看时使用。"},
        {"name": "search_order(app, search_info, order_status)", "description": "在app应用程序中搜索订单"},
        {"name": "open_invoice_page(app, page_type)", "description": "在app应用程序中打开与发票相关的页面"},
        {"name": "open_cart_content(app, filter_type)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）指定类型的商品"},
        {"name": "search_cart_content(app, search_info)", "description": "在app应用程序中查看购物车/采购车（阿里巴巴的叫法）查找商品"},
        {"name": "open_customer_service(app)", "description": "在app应用程序中联系客服"},
        {"name": "sign_in(app, page_type)", "description": "在app程序中完成每日签到，领取积分、金币等奖励的操作"},
        {"name": "open_favorite_goods(app, filter_type, order_type)", "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面，并按照条件进行筛选"},
        {"name": "open_favorite_stores(app, filter_type)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并按照条件进行筛选"},
        {"name": "search_in_favorite_goods(app, search_info)", "description": "在app程序中打开收藏的、喜爱、想要或关注商品的页面，并在其中的搜索栏中进行搜索"},
        {"name": "search_in_favorite_stores(app, search_info)", "description": "在app程序中打开收藏的喜爱或关注店铺的页面，并在其中的搜索栏搜索商品"},
        {"name": "search_goods(app, search_info, order_type)", "description": "在app程序中依据名称搜索商品，可以指定搜索结果的排序方式"},
        {"name": "search_stores(app, search_info, filter_type, order_type)", "description": "在app程序中依据名称搜索店铺，可以使用筛选器限制搜索结果，也可以指定搜索结果的排序方式"},
        {"name": "open_search_history(app)", "description": "打开app程序的搜索历史界面"},
        {"name": "delete_search_history(app)", "description": "清除app中的搜索历史"},
        {"name": "open_camera_search(app)", "description": "打开app程序的图片搜索功能"},
        {"name": "open_logistics_receive(app, filter_type)", "description": "打开显示已购商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_logistics_send(app, filter_type)", "description": "打开显示已售商品信息的界面，查看相关物流信息，并根据物流情况进行筛选"},
        {"name": "open_express_delivery(app)", "description": "打开app寄送快递的界面"},
        {"name": "open_app(app)", "description": "打开指定的应用程序"},
    ]
    return tools

# ==============================================================================
# 区域 2: 辅助函数 (来自原脚本2)
# ==============================================================================
def create_instruction_to_tool_map(mapping_filename: str) -> dict:
    """
    从CSV文件解析工具映射表，创建一个从 instruction_template 到 function_name 的字典。
    Args:
        mapping_filename: 包含工具映射关系的CSV文件名。
    Returns:
        一个字典，键是 instruction_template, 值是对应的 function_name。
    """
    mapping_dict = {}
    print(f"正在读取工具映射文件: {mapping_filename}...")
    try:
        with open(mapping_filename, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                function_name = row.get('function_name', '').strip()
                instructions_str = row.get('包含的指令', '')

                if not function_name:
                    continue

                # 特殊处理 '启动' 类别
                if function_name == 'open_app(app)' and not instructions_str:
                    app_name = row.get('app', '').strip()
                    if app_name:
                        instruction_template = f"打开{app_name}"
                        mapping_dict[instruction_template] = function_name
                    continue

                # 解析指令列表
                if instructions_str:
                    try:
                        instructions = json.loads(instructions_str)
                        for instruction in instructions:
                            mapping_dict[instruction.strip()] = function_name
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析JSON指令列表 '{instructions_str}'。跳过此条目。", file=sys.stderr)
                        continue
    
    except FileNotFoundError:
        print(f"错误: 映射文件 '{mapping_filename}' 未找到。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"读取或解析映射文件时发生未知错误: {e}", file=sys.stderr)
        return None
        
    print(f"工具映射表成功创建，共包含 {len(mapping_dict)} 条指令映射。")
    return mapping_dict

# ==============================================================================
# 区域 3: 主逻辑程序
# ==============================================================================
def main():
    """
    主执行函数，整合所有步骤。
    """
    # --- 1. 配置区域 ---
    data_file_path = r'D:\Agent\data\全部生成数据表格 - 0707-500条指令-各5个query.csv'
    mapping_file_path = r'D:\Agent\data\大类-工具映射关系表-0815-Cleaned.csv'
    output_file_path = r'D:\Agent\data\单gt.csv' # 最终输出文件

    # --- 2. 准备工作：加载工具定义和映射关系 ---
    print("--- 步骤 1: 准备工具定义和映射 ---")
    
    # 从 get_exact_tool_definitions 获取所有工具的详细信息
    all_tools_dict = {tool['name']: tool for tool in get_exact_tool_definitions()}
    print(f"   - 全局工具字典构建完成，共加载 {len(all_tools_dict)} 个工具。")
    
    # 从映射文件创建 instruction_template -> function_name 的字典
    instruction_tool_map = create_instruction_to_tool_map(mapping_file_path)
    if instruction_tool_map is None:
        print("程序因无法创建指令映射而中止。", file=sys.stderr)
        return
    print("-" * 30)

    # --- 3. 读取原始数据 ---
    print(f"--- 步骤 2: 读取原始数据文件 '{os.path.basename(data_file_path)}' ---")
    try:
        required_columns = ['category', 'app_name', 'instruction_template', 'final_query', 'is_train']
        df = pd.read_csv(data_file_path, usecols=required_columns)
        print(f"   - 文件读取成功，共 {len(df)} 行。\n")
    except Exception as e:
        print(f"错误: 读取文件 '{data_file_path}' 失败。 {e}", file=sys.stderr)
        return
    print("-" * 30)

    # --- 4. 定义核心处理函数 ---
    def get_single_ground_truth_tool(row):
        """
        根据行数据中的指令模板，查找并返回唯一的、正确的工具定义。
        """
        instruction_template = row['instruction_template'].strip()
        
        # 步骤 A: 根据指令模板，从映射中找到正确的工具名称
        correct_tool_name = instruction_tool_map.get(instruction_template)
        
        if not correct_tool_name:
            print(f"警告: 在第 {row.name + 2} 行, 未找到指令 '{instruction_template}' 的映射。将生成空工具列表。", file=sys.stderr)
            return '[]' # 返回空JSON列表作为回退

        # 步骤 B: 根据工具名称，从全局工具字典中找到完整的工具定义
        correct_tool_definition = all_tools_dict.get(correct_tool_name)

        if not correct_tool_definition:
            print(f"警告: 在第 {row.name + 2} 行, 指令 '{instruction_template}' 对应的工具 '{correct_tool_name}' 在全局工具定义中未找到。将生成空工具列表。", file=sys.stderr)
            return '[]' # 返回空JSON列表作为回退

        # 步骤 C: 将找到的单个工具打包成列表，并转换为JSON字符串
        return json.dumps([correct_tool_definition], ensure_ascii=False)

    # --- 5. 应用函数，直接生成最终的 'available_tools' 列 ---
    print("--- 步骤 3: 为每一行匹配唯一的 Ground Truth 工具 ---")
    df['available_tools'] = df.apply(get_single_ground_truth_tool, axis=1)
    print("   - 工具匹配完成！\n")
    print("-" * 30)

    # --- 6. 保存最终结果到目标文件 ---
    print(f"--- 步骤 4: 保存最终结果到 '{os.path.basename(output_file_path)}' ---")
    try:
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"   - 文件已成功保存到: {output_file_path}\n")
    except Exception as e:
        print(f"错误: 无法写入输出文件 '{output_file_path}': {e}", file=sys.stderr)
        return
    print("-" * 30)
    
    # --- 7. 打印结果预览 ---
    print("🎉 处理完成，最终结果预览 (前5行)：\n")
    print(df.head().to_string())
    print("-" * 30)

# ==============================================================================
# 区域 4: 程序入口
# ==============================================================================
if __name__ == "__main__":
    main()