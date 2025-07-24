import pandas as pd
import json
import ast
from tqdm import tqdm

def get_exact_tool_definitions():
    """
    权威的工具定义库。
    这是所有工具及其描述的唯一真实来源。
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

def create_ground_truth_map(mapping_df: pd.DataFrame, tool_definitions: list) -> dict:
    """
    根据映射表和工具定义，创建一个从 instruction_template 到完整工具定义(JSON字符串)的映射字典。
    """
    print("--- 正在创建指令到工具的映射... ---")
    
    # 1. 创建一个从工具名到完整定义的字典，方便查找
    tool_def_map = {tool['name']: tool for tool in tool_definitions}
    
    # 2. 创建最终的映射字典
    instruction_to_tool_map = {}
    
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="处理映射表"):
        # 清理 function_name 中可能存在的首尾空格
        tool_name = row['function_name'].strip()
        
        # 检查工具是否存在于我们的权威定义中
        if tool_name not in tool_def_map:
            print(f"警告: 映射表中的工具 '{tool_name}' 不在权威工具定义中，将被忽略。")
            continue
            
        # 解析 "包含的指令" 列，它是一个字符串形式的列表
        try:
            instructions = ast.literal_eval(row['包含的指令'])
            if not isinstance(instructions, list):
                instructions = [str(instructions)]
        except (ValueError, SyntaxError):
            print(f"警告: 无法解析指令 '{row['包含的指令']}'，跳过此行。")
            continue
            
        # 获取完整的工具定义
        full_tool_def = tool_def_map[tool_name]
        
        # 将工具定义格式化为所需的列表包含字典的结构，并转为JSON字符串
        # 你的原始代码期望的是一个列表，所以我们把它放在列表里
        tool_json_string = json.dumps([full_tool_def], ensure_ascii=False)
        
        # 为每个指令创建映射
        for instruction in instructions:
            instruction_to_tool_map[instruction] = tool_json_string
            
    print(f"映射创建完成，共处理 {len(instruction_to_tool_map)} 条唯一的指令模板。\n")
    return instruction_to_tool_map

def main():
    # --- 文件路径配置 ---
    # 你的原始数据文件
    original_data_path = r'/home/workspace/lgq/shop/data/单gt.csv' 
    # 你的映射表文件
    mapping_table_path = r'/home/workspace/lgq/shop/data/大类-工具映射关系表-0707-Cleaned.csv' 
    # 清洗后要保存的新数据文件
    corrected_data_path = r'/home/workspace/lgq/shop/data/corrected_data.csv'

    # --- 开始处理 ---
    print("--- 步骤 1: 加载文件 ---")
    try:
        data_df = pd.read_csv(original_data_path)
        mapping_df = pd.read_csv(mapping_table_path)
        print(f"成功加载原始数据 {len(data_df)} 条，映射表 {len(mapping_df)} 条。\n")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        print("请确保 '单gt.csv' 和 'mapping_table.csv' 文件存在于正确的位置。")
        return

    # --- 步骤 2: 创建映射并修正数据 ---
    all_tool_defs = get_exact_tool_definitions()
    instruction_map = create_ground_truth_map(mapping_df, all_tool_defs)
    
    print("--- 步骤 3: 修正 'available_tools' 列 ---")
    
    # 使用 .map() 方法高效地替换列内容
    # 如果原始指令在映射中找不到，则填充为一个空列表的JSON字符串 '[]'
    data_df['available_tools'] = data_df['instruction_template'].map(instruction_map).fillna('[]')
    
    # 统计修正结果
    corrected_count = (data_df['available_tools'] != '[]').sum()
    print(f"修正完成！共 {corrected_count} / {len(data_df)} 条数据的工具标签被成功替换。\n")

    # --- 步骤 4: 保存结果 ---
    try:
        # 使用 utf-8-sig 编码以确保 Excel 等软件能正确打开包含中文的CSV文件
        data_df.to_csv(corrected_data_path, index=False, encoding='utf-8-sig')
        print(f"🎉 成功！已将修正后的数据保存到: {corrected_data_path}")
    except Exception as e:
        print(f"错误: 保存文件失败 - {e}")

if __name__ == "__main__":
    main()