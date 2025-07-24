import pandas as pd
import json
import os
import sys

# ==============================================================================
# 区域 1: 815版本最新工具定义 (保持不变)
# ==============================================================================
def get_exact_tool_definitions():
    """
    定义了系统中所有可用的工具及其完整的函数签名和描述。
    这是根据《815版本指令集对应的工具定义初稿-0715》更新的完整工具列表。
    """
    tools = [
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

# ==============================================================================
# 区域 2: 从CSV文件加载指令映射 (保持不变)
# ==============================================================================
def load_instruction_map_from_csv(mapping_filename: str) -> dict:
    """
    从CSV文件解析工具映射表,创建一个从 instruction_template 到 function_name 的字典。
    """
    mapping_dict = {}
    print(f"正在读取工具映射文件: {mapping_filename}...")
    try:
        map_df = pd.read_csv(mapping_filename, usecols=['function_name', '包含的指令'])
        
        for _, row in map_df.iterrows():
            function_name = row.get('function_name', '').strip()
            instructions_str = row.get('包含的指令', '')

            if not function_name or pd.isna(instructions_str):
                continue
            
            try:
                instructions = json.loads(instructions_str)
                for instruction in instructions:
                    mapping_dict[instruction.strip()] = function_name
            except (json.JSONDecodeError, TypeError):
                continue
    
    except FileNotFoundError:
        print(f"错误: 映射文件 '{mapping_filename}' 未找到。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"读取或解析映射文件 '{mapping_filename}' 时发生未知错误: {e}", file=sys.stderr)
        return None
        
    print(f"工具映射表成功创建，共包含 {len(mapping_dict)} 条指令映射。")
    return mapping_dict

# ==============================================================================
# 区域 3: 主逻辑程序 (已适配新列)
# ==============================================================================
def main():
    """
    主执行函数，整合所有步骤。
    """
    # --- 1. 配置区域 ---
    data_file_path = r'D:\Agent\data\二期指令生成的数据集 - 单轮-单任务.csv'
    mapping_file_path = r'D:\Agent\data\大类-工具映射关系表-0815-Cleaned.csv'
    output_file_path = r'D:\Agent\code\bm25_recall\data\single_gt_output_with_plan.csv'
    
    # --- 2. 准备工作：加载工具定义和映射关系 ---
    print("--- 步骤 1: 准备工具定义和指令映射 ---")
    
    all_tools_list = get_exact_tool_definitions()
    all_tools_dict = {tool['name']: tool for tool in all_tools_list}
    print(f"   - 全局工具字典构建完成，共加载 {len(all_tools_dict)} 个工具。")
    
    instruction_tool_map = load_instruction_map_from_csv(mapping_file_path)
    if instruction_tool_map is None:
        print("程序因无法创建指令映射而中止。", file=sys.stderr)
        return
    print("-" * 30)

    # --- 3. 读取原始数据 ---
    print(f"--- 步骤 2: 读取新数据文件 '{os.path.basename(data_file_path)}' ---")
    try:
        # 【修改点 1】在 usecols 中增加 'plan（在xx中做什么）'
        required_cols = ['大类', 'app', '指令', 'query', 'plan（在xx中做什么）']
        df = pd.read_csv(data_file_path, usecols=required_cols)
        print(f"   - 文件读取成功，共 {len(df)} 行。\n")
    except ValueError as e:
        print(f"错误: 读取文件 '{data_file_path}' 失败。请确保文件中包含以下所有列: {required_cols}。错误详情: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}", file=sys.stderr)
        return
    print("-" * 30)

    # --- 4. 定义核心处理函数 ---
    def get_single_ground_truth_tool(row):
        """
        根据行数据中的 '指令' 列，查找并返回唯一的、正确的工具定义。
        """
        instruction_value = row['指令']
        
        if pd.isna(instruction_value) or not isinstance(instruction_value, str):
            return '[]'

        instruction_template = instruction_value.strip()
        
        correct_tool_signature = instruction_tool_map.get(instruction_template)
        
        if not correct_tool_signature:
            return '[]'

        correct_tool_definition = all_tools_dict.get(correct_tool_signature)

        if not correct_tool_definition:
            return '[]'

        return json.dumps([correct_tool_definition], ensure_ascii=False)

    # --- 5. 应用函数，生成 'ground_truth_tool' 列 ---
    print("--- 步骤 3: 为每一行匹配唯一的 Ground Truth 工具 ---")
    df['ground_truth_tool'] = df.apply(get_single_ground_truth_tool, axis=1)
    success_count = df[df['ground_truth_tool'] != '[]'].shape[0]
    print(f"   - 工具匹配完成！成功匹配 {success_count} / {len(df)} 行。")
    print("-" * 30)

    # --- 6. 保存最终结果到目标文件 ---
    print(f"--- 步骤 4: 保存最终结果到 '{os.path.basename(output_file_path)}' ---")
    try:
        # 【修改点 2】在输出列中增加 'plan（在xx中做什么）' 并调整顺序
        output_cols = ['大类', 'app', 'query', 'plan（在xx中做什么）', '指令', 'ground_truth_tool']
        output_df = df[output_cols]
        output_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"   - 文件已成功保存到: {output_file_path}\n")
    except Exception as e:
        print(f"错误: 无法写入输出文件 '{output_file_path}': {e}", file=sys.stderr)
        return
    
    # --- 7. 打印结果预览 ---
    print("🎉 处理完成，最终结果预览 (前5行)：\n")
    with pd.option_context('display.max_colwidth', 100):
        print(output_df.head().to_string())
    print("-" * 30)


if __name__ == "__main__":
    main()