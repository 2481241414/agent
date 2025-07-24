import pandas as pd
import os
import json # 用于将列表/字典转为格式化的字符串

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★  核心模块：使用完整的函数签名作为工具名称  ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def get_exact_tool_definitions():

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
    # 请确保这里的列表包含了您映射表中所有的 function_name
    return tools

# --- 主程序开始 ---

# --- 配置区域 ---
data_file_path = r'D:\Agent\data\全部生成数据表格 - 0707-500条指令-各5个query.csv'
mapping_file_path = r'D:\Agent\data\大类-工具映射关系表-0707-Cleaned.csv'
# 使用新的输出文件名，避免混淆
output_file_path = r'D:\Agent\data\corrected_data.csv' 

# --- 代码正文 ---

# 步骤 1: 从文件中读取数据
print("1. 正在从文件中读取数据...")
try:
    required_columns = ['category', 'app_name', 'instruction_template', 'final_query', 'is_train']
    data_df = pd.read_csv(data_file_path, usecols=required_columns)
    mapping_df = pd.read_csv(mapping_file_path)
    print(f"   - '{os.path.basename(data_file_path)}' 读取成功 (只加载了指定列)")
    print(f"   - '{os.path.basename(mapping_file_path)}' 读取成功\n")
except Exception as e:
    print(f"错误：读取文件失败。 {e}")
    exit()

# 步骤 2: 构建所有工具的字典 (使用完整的函数签名作为键)
print("2. 正在构建 all_tools_dict (使用完整的函数签名作为键)...")
exact_tools_list = get_exact_tool_definitions()
all_tools_dict = {tool['name']: tool for tool in exact_tools_list}
print(f"   - 工具字典构建完成，共加载 {len(all_tools_dict)} 个精确定义的工具。\n")


# 步骤 3: 定义工具过滤函数 (函数逻辑不变)
def filter_tools(app_name: str, category: str, mapping_df: pd.DataFrame, all_tools_dict: dict) -> list:
    """
    根据 app_name 和 category 从映射表中筛选出可用的工具。
    """
    filtered_df = mapping_df[
        (mapping_df['app'] == app_name) &
        (mapping_df['大类'] == category)
    ]
    tool_names = filtered_df['function_name'].unique().tolist()
    
    available_tools = []
    for name in tool_names:
        if name in all_tools_dict:
            available_tools.append(all_tools_dict[name])
        else:
            print(f"警告：工具 '{name}' 在映射表中存在，但在 get_exact_tool_definitions() 中未找到。请检查并补充定义。")
            
    return available_tools
print("3. 工具过滤函数已定义\n")


# 步骤 4: 应用函数，为 data_df 生成新列 'available_tools'
print("4. 正在为每一行数据匹配可用工具...")
data_df['available_tools'] = data_df.apply(
    lambda row: filter_tools(
        app_name=row['app_name'],
        category=row['category'],
        mapping_df=mapping_df,
        all_tools_dict=all_tools_dict
    ),
    axis=1
)
print("   - 工具匹配完成！\n")


# 步骤 5: 将最终结果保存到新的 CSV 文件
print(f"5. 正在将结果保存到文件 '{os.path.basename(output_file_path)}'...")
data_df['available_tools'] = data_df['available_tools'].apply(
    lambda x: json.dumps(x, ensure_ascii=False) if x else '[]'
)
data_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
print(f"   - 文件已成功保存到: {output_file_path}\n")


# 步骤 6: 在控制台打印最终结果的预览
print("-" * 30)
print("🎉 处理完成，最终结果预览 (前5行)：\n")
print(data_df.head().to_string())
print("-" * 30)