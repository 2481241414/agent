import pandas as pd
import json
import os

def get_exact_tool_definitions():
    """
    返回所有工具的完整定义列表。
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

def create_tool_mapping(tools):
    """
    根据工具定义列表，创建一个从工具简称到完整JSON字符串的映射。
    """
    tool_map = {}
    for tool in tools:
        short_name = tool['name'].split('(')[0]
        # ensure_ascii=False 确保中文字符在生成JSON时不被转义
        json_string = json.dumps([tool], ensure_ascii=False)
        tool_map[short_name] = json_string
    return tool_map

def run_data_processing(data1_path, data2_path, output_path):
    """
    读取并处理两份CSV数据，然后将结果保存到新的CSV文件。

    :param data1_path: 第一份数据的文件路径 (CSV)
    :param data2_path: 第二份数据的文件路径 (CSV)
    :param output_path: 处理完成后输出的文件路径 (CSV)
    """
    # 检查输入文件是否存在
    if not os.path.exists(data1_path):
        print(f"错误: 输入文件 '{data1_path}' 未找到。")
        return
    if not os.path.exists(data2_path):
        print(f"错误: 输入文件 '{data2_path}' 未找到。")
        return

    print("开始处理...")

    # 1. 创建工具简称到完整定义的映射
    all_tools = get_exact_tool_definitions()
    tool_map = create_tool_mapping(all_tools)
    print("步骤 1/5: 工具定义映射创建完成。")

    # 2. 读取第一份数据
    df1 = pd.read_csv(data1_path)
    print(f"步骤 2/5: 成功读取 '{data1_path}' ({len(df1)} 行)。")

    # 3. 扩充第一份数据的 ground_truth 列
    # 使用 .get(key, key) 如果找不到映射，则保留原值，避免出错
    df1['ground_truth_expanded'] = df1['ground_truth'].apply(lambda gt: tool_map.get(str(gt), str(gt)))
    print("步骤 3/5: 第一份数据的 'ground_truth' 列已扩充。")

    # 4. 创建从 query 到扩充后 ground_truth 的映射字典
    # 去除重复的 'query'，保留第一个出现的，以避免映射冲突
    df1_unique_query = df1.drop_duplicates(subset=['query'])
    query_to_tool_map = df1_unique_query.set_index('query')['ground_truth_expanded'].to_dict()
    print("步骤 4/5: 'query' 到工具定义的映射创建完成。")

    # 5. 读取并处理第二份数据
    df2 = pd.read_csv(data2_path)
    print(f"步骤 5/5: 成功读取 '{data2_path}' ({len(df2)} 行)，开始更新 'ground_truth_tool' 列...")

    # 定义更新函数，用于填充空缺值
    def update_tool(row):
        # 检查'ground_truth_tool'是否为空的标记'[]'
        if row['ground_truth_tool'] == '[]':
            # 从映射中查找对应的工具。如果找不到，则返回原值 '[]'
            return query_to_tool_map.get(row['query'], '[]')
        return row['ground_truth_tool']

    # 应用更新函数
    df2['ground_truth_tool'] = df2.apply(update_tool, axis=1)

    # 6. 保存结果
    # 使用 encoding='utf-8-sig' 来确保在Excel中打开时不会出现中文乱码
    df2.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n处理完成！")
    print(f"最终结果已保存至: '{output_path}'")

# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 配置 ---
    # 请将 'data1.csv' 和 'data2.csv' 替换为您的实际文件名
    first_data_file = '/home/workspace/lgq/shop/data/hybrid_recall_results_api_测试_4b_标注.csv'
    second_data_file = '/home/workspace/lgq/shop/data/single_gt_购物语料-测试结果标注 - 场景单任务1_0815.csv'
    
    # 设置输出文件名
    output_file = '/home/workspace/lgq/shop/data/single_gt_购物语料-测试结果标注 - 场景单任务1_0815_processed_data.csv'
    
    # --- 运行 ---
    run_data_processing(
        data1_path=first_data_file,
        data2_path=second_data_file,
        output_path=output_file
    )