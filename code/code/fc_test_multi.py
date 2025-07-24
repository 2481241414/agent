import os
import sys
import json
import time
import pandas as pd
from openai import OpenAI

# --- API Key 和客户端初始化 ---
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key is None:
    print("警告：未在环境变量中找到 DASHSCOPE_API_KEY，将使用代码中硬编码的Key。", file=sys.stderr)
    api_key = "sk-4fcc85e2509649198bdcafa4e985ce6e"

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --- 模型调用函数 ---
def get_qwen_response(instruction: str, user_input: str, model: str = "qwen3-32b") -> str:
    try:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
        ]
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"enable_thinking": False},
        )
        end_time = time.time()
        latency = end_time - start_time
        print(f"    API 调用耗时: {latency:.4f} 秒")
        return completion.choices[0].message.content
    except Exception as e:
        error_message = f"调用API时出错: {e}"
        print(error_message, file=sys.stderr)
        return json.dumps({"error": str(e)})

# --- 数据处理和工具过滤函数 ---
def load_and_process_tool_mapping(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['大类'] = df['大类'].str.strip()
        df['app'] = df['app'].str.strip()
        df['tool_name'] = df['function_name'].str.strip().str.extract(r'(\w+)', expand=False)
        df.dropna(subset=['大类', 'app', 'tool_name'], inplace=True)
        print(f"✅ 成功从 '{filepath}' 加载并处理了 {len(df)} 条工具映射关系。")
        return df[['大类', 'app', 'tool_name']]
    except FileNotFoundError:
        print(f"错误：找不到工具映射文件 '{filepath}'。请确保该文件存在且路径正确。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"处理工具映射文件时出错: {e}", file=sys.stderr)
        sys.exit(1)

def filter_tools(app_name: str, category: str, mapping_df: pd.DataFrame, all_tools_dict: dict) -> list:
    filtered_df = mapping_df[
        (mapping_df['app'] == app_name) &
        (mapping_df['大类'] == category)
    ]
    tool_names = filtered_df['tool_name'].unique().tolist()
    # print(tool_names)
    available_tools = [all_tools_dict[name] for name in tool_names if name in all_tools_dict]
    print(f"    🔍 过滤后的可用工具: {[tool['name'] for tool in available_tools]}")
    return available_tools

# --- 评测函数 ---
# --- 评测函数 ---
def evaluate_prediction(prediction_str: str, label_str: str) -> dict:
    try:
        # 核心修正：在进行任何操作前，先将label_str强制转换为字符串。
        # 这样即使读入的是NaN(float)，也会被转成字符串"nan"，不会报错。
        label_str = str(label_str) 
        
        # 预处理字符串，将Python的None转为JSON的null，以便解析
        prediction_str_fixed = prediction_str.replace("None", "null")
        label_str_fixed = label_str.replace("None", "null")
        
        prediction_obj = json.loads(prediction_str_fixed)
        label_obj = json.loads(label_str_fixed)
        
        is_correct = (prediction_obj == label_obj)
        tool_name_correct = (prediction_obj.get('tool_name') == label_obj.get('tool_name'))
        params_correct = (prediction_obj.get('tool_parameters') == label_obj.get('tool_parameters'))
        
        return {
            "is_correct": is_correct,
            "tool_name_correct": tool_name_correct,
            "params_correct": params_correct,
            "prediction": prediction_obj,
            "label": label_obj
        }
    except (json.JSONDecodeError, TypeError):
        # 如果label是"nan"或其他非json字符串，这里会捕获错误，判定为不正确，符合预期。
        return {
            "is_correct": False,
            "tool_name_correct": False,
            "params_correct": False,
            "prediction": prediction_str,
            "label": label_str, # 保留原始的、可能有问题的label，方便调试
            "error": "JSONDecodeError or TypeError"
        }

# --- 主程序 ---
if __name__ == "__main__":
    model_type = "qwen-coder-turbo"
    overall_start_time = time.time()

    # --- 1. 定义所有工具和Prompt ---
    instruction = """
# 角色
你是一个AI工具调用专家。你的任务是根据用户输入，选择最合适的工具并填充参数，然后以JSON格式输出。

# 输入格式
你会收到一个JSON对象，包含四个键：
1. `app`: (string) 应用名称。
2. `category`: (string) 功能分类，用于快速筛选工具。
3. `user_query`: (string) 用户的原始自然语言请求。
4. `available_tools`: (array) 本次任务可用的工具列表。

# 核心指令：你必须严格遵循以下四步流程

1.  **理解意图 (Understand Intent)**
    *   **优先使用 `category` 进行初步筛选**。
    *   结合 `user_query` 和工具的 `description` 理解用户的具体动作和需求，包括工具适用的App范围。

2.  **选择工具 (Select Tool)**
    *   在 `available_tools` 列表中，选择一个最匹配用户意图的工具。

3.  **验证与提取参数 (Validate & Extract Parameters)**
    *   对于选择的工具，从 `user_query` 中提取所有需要的参数值。
    *   **验证**：检查提取出的参数值是否存在于该工具参数的`description`中为当前`app`列出的可用值列表里。
    *   **处理默认值**：如果某个参数是可选的，并且`user_query`中没有提及任何该参数的有效值，你【必须】使用该参数在`description`中指定的默认值（通常是`"无"`）来填充。如果描述中指明可以为 `null`，你就填充 `null`。

4.  **填充参数或返回错误 (Fill or Error)**
    *   **如果工具匹配且所有参数都有效（或已正确填充默认值）**：
        *   **`app` 参数**：【必须】直接使用输入中提供的 `app` 值。
        *   **其他参数**：使用上一步提取或确定的值进行填充。
    *   **如果出现以下任一情况，请严格输出错误JSON**：
        *   `available_tools` 列表为空。
        *   没有工具能匹配用户的意图。
        *   用户请求的参数值无效（即不在工具定义的支持范围内）。
        *   错误格式为：{"error": "No valid tool or parameter found to fulfill the request"}。

# 输出格式
*   最终输出【必须】是一个单一、严格的JSON对象。
*   【禁止】包含任何解释、注释或Markdown标记 (如 ```json ... ```)。
*   JSON结构必须为: {"tool_name": "...", "tool_parameters": { ... } } 或 {"error": "..."}。

# 示例 (学习这个思考过程和输出)

---
### 示例 1: 成功匹配并填充明确参数

#### 输入:
{"app": "淘宝", "category": "订单", "user_query": "帮我看看淘宝待发货的订单", "available_tools": [{"name": "open_orders", ...}]}
#### 思考:
1.  意图是查看订单，工具`open_orders`匹配。
2.  参数`order_status`在`user_query`中明确为"待发货"。
3.  "待发货"是淘宝支持的有效值。
4.  正常填充。
#### 输出:
{"tool_name": "open_orders", "tool_parameters": {"app": "淘宝", "order_status": "待发货"}}

---
### 示例 2: 无可用工具的情况

#### 输入:
{"app": "微信", "category": "订单", "user_query": "看看我在微信的订单", "available_tools": []}
#### 思考:
1.  `available_tools`为空。
2.  无法满足请求，返回错误。
#### 输出:
{"error": "No valid tool or parameter found to fulfill the request"}

---
### 示例 3: 参数值不在候选列表（无效参数）的情况

#### 输入:
{"app": "京东", "category": "搜索", "user_query": "用京东搜一下手机，按人气排序", "available_tools": [{"name": "search_goods", "description": "...", "inputSchema": {"properties": {"order_type": {"description": "默认值: '无'。京东支持:['无','综合','销量','价格从低到高','价格从高到低','按评论数从高到低']"}}}}}]}
#### 思考:
1.  意图是搜索商品，工具`search_goods`匹配。
2.  用户请求的`order_type`是"人气"。
3.  检查`search_goods`工具对京东的支持列表，发现`order_type`中不包含"人气"。
4.  参数值无效，无法满足请求，返回错误。
#### 输出:
{"error": "No valid tool or parameter found to fulfill the request"}

---
### 示例 4: 使用 `default` 值填充参数

#### 输入:
{"app": "抖音", "category": "订单", "user_query": "在抖音中查看我的订单", "available_tools": [{"name": "open_orders", "description": "...", "inputSchema": {"properties": {"order_status": {"description": "订单状态。默认值: '无'。抖音支持:['无','待支付',...]"}}}}}]}
#### 思考:
1.  意图是查看订单，工具`open_orders`匹配。
2.  `user_query`中未指定`order_status`。
3.  工具定义`order_status`参数的默认值为`'无'`。
4.  使用默认值填充。
#### 输出:
{"tool_name": "open_orders", "tool_parameters": {"app": "抖音", "order_status": "无"}}
"""

    # ==============================================================================
    # ==                                  工具列表                                 ==
    # ==============================================================================
    all_tools_list = [
        # --- 订单 ---
        {"name": "open_orders", "description": "在app中查看指定状态的订单列表 (该函数不涉及闲鱼、转转)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "order_status": {"type": "string", "description": "订单状态。默认值: '无'。阿里巴巴支持:['无','待付款','待发货','待收货','待评价','退款-售后'], 淘宝支持:['无','待付款','待发货','待收货','待评价','退款/售后'], 天猫支持:['无','待付款','待发货','待收货','待评价','退款和售后'], 抖音/抖音火山版/抖音极速版/快手/快手极速版支持:['无','待支付','待发货','待收货','待评价','售后'], 得物支持:['无','待付款','待发货','待收货'], 京东支持:['无','待付款','待收货','待服务','待评价','退款/售后'], 拼多多支持:['无','待付款','待收货','待分享','待发货','退款售后'], 唯品会支持:['无','已取消的待付款','已完成的待付款','待付款','待评价','待收货','退换售后']"}}, "required": ["app", "order_status"]}},
        {"name": "open_second_hand_orders", "description": "在二手交易中查看指定状态的订单列表 (该函数不涉及其他12个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称，仅限闲鱼、转转"}, "order_status": {"type": "string", "description": "订单状态。默认值: '无'。闲鱼支持:['无','全部','待付款','待发货','待收货','待评价','退款中'], 转转支持:['无','待付款','待发货','待收货','待评价','退款售后','未完结的退款售后','待评价的退款售后']"}, "product_status": {"type": "string", "description": "用户售卖商品的状态。默认值: '无'。闲鱼支持:['无','在卖','草稿','已下架'], 转转支持:['无','进行中的','已完成的','已关闭的']"}, "transaction_type": {"type": "string", "description": "区分用户的状态是买入还是卖出。闲鱼支持:['我买到的','我发布的','我卖出的'], 转转支持:['我发布的','我卖的','我购买的']"}}, "required": ["app", "order_status", "product_status", "transaction_type"]}},
        {"name": "search_order", "description": "在app中搜索指定内容的订单", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "search_info": {"type": "string", "description": "搜索内容"}, "order_status": {"type": "string", "description": "对订单的限定条件。默认值: '无'。唯品会支持:['无','退换售后'], 闲鱼支持:['我发布的','我卖出的','我买到的'], 转转支持:['我购买的','我卖出的'], 其他11个App支持:['无']"}}, "required": ["app", "search_info", "order_status"]}},
        {"name": "open_reviews", "description": "在app中查看我的评价 (该函数只涉及拼多多)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称，仅限拼多多"}}, "required": ["app"]}},
        {"name": "apply_after_sales", "description": "申请退款或售后 (该函数只涉及得物)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称，仅限得物"}}, "required": ["app"]}},

        # --- 发票 ---
        {"name": "open_invoice_page", "description": "查看发票相关服务页面 (该函数不涉及其他9个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "page_type": {"type": "string", "description": "打开的页面类型。得物支持:['发票服务'], 京东支持:['发票抬头管理','全部发票','换开/合开发票','发票专票提交','发票申请记录'], 拼多多支持:['全部发票','已开票发票','申请中发票','待申请发票'], 淘宝支持:['发票中心','申请中发票','已开票发票','未申请发票'], 唯品会支持:['发票服务','开具发票','我的发票']"}}, "required": ["app", "page_type"]}},

        # --- 购物车 ---
        {"name": "open_cart_content", "description": "查看购物车中指定类型的商品 (该函数不涉及闲鱼、转转、拼多多)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称。阿里巴巴叫'采购车', 得物叫'想要的商品', 其他叫'购物车'"}, "filter_type": {"type": "string", "description": "指定的商品类型。默认值: '无'。阿里巴巴支持:['无','常购','代发','现货'], 得物支持:['无','降价'], 抖音支持:['无','有货','降价'], 抖音火山版支持:['无','降价'], 抖音极速版支持:['无','有货','降价'], 京东支持:['无','送礼','凑单','跨店满减','降价'], 快手/快手极速版/天猫支持:['无'], 淘宝支持:['无','降价','常购','失效'], 唯品会支持:['无','近期加购','降价','有货']"}}, "required": ["app", "filter_type"]}},
        {"name": "search_cart_content", "description": "在购物车中搜索指定物品 (该函数只涉及抖音、抖音极速版、京东、淘宝、唯品会)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "search_info": {"type": "string", "description": "搜索内容"}}, "required": ["app", "search_info"]}},
        {"name": "open_cart_page", "description": "打开购物车指定界面 (该函数只涉及淘宝)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称，仅限淘宝"}, "page_type": {"type": "string", "description": "购物车中的功能界面。淘宝支持:['管理','分组']"}}, "required": ["app", "page_type"]}},
        
        # --- 客服 ---
        {"name": "open_customer_service", "description": "在app应用程序中联系官方客服 (该函数涉及全部14个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}},

        # --- 签到 ---
        {"name": "sign_in", "description": "在app中(具体某个子页面)使用签到功能 (该函数不涉及转转)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "page_type": {"type": "string", "description": "具体某个子页面的签到功能。默认值: '无'。抖音/快手/快手极速版/拼多多/京东/闲鱼/抖音火山版/阿里巴巴/唯品会/得物支持:['无'], 抖音极速版支持:['赚钱','我的钱包'], 淘宝支持:['淘金币','红包'], 天猫支持:['领现金','红包']"}}, "required": ["app", "page_type"]}},

        # --- 收藏 ---
        {"name": "open_favorite_goods", "description": "打开商品收藏夹，并且使用筛选条件进行筛选 (该函数不涉及得物、转转)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "filter_type": {"type": "string", "description": "商品收藏夹的筛选条件。默认值: '无'。抖音/抖音极速版/快手/快手极速版/拼多多支持:['无'], 淘宝支持:['无','有降价','已买过','低库存','已失效','7天内','30天内','90天内','半年前','一年前'], 京东支持:['无','降价','促销','有货','下架'], 天猫支持:['无','降价','已买过','低库存','已失效','7天内','30天内','90天内','半年前','一年前'], 闲鱼支持:['无','降价','有效','失效'], 阿里巴巴支持:['无','降价','低价补货','买过'], 唯品会支持:['无','降价','有货','已失效','7天内','1个月内','3个月内','半年前']"}, "order_type": {"type": "string", "description": "商品排列方式。默认值: '无'。淘宝支持:['无','最近收藏在前','最早收藏在前'], 天猫支持:['无','最近收藏在前','最早收藏在前'], 其他支持的应用均为:['无']"}}, "required": ["app", "filter_type", "order_type"]}},
        {"name": "open_favorite_stores", "description": "打开店铺收藏夹，并且使用筛选条件进行筛选 (该函数不涉及其他9个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "filter_type": {"type": "string", "description": "店铺收藏夹的筛选条件。默认值: '无'。淘宝支持:['无','特别关注','直播中','有上新'], 拼多多/京东/天猫/阿里巴巴支持:['无']"}}, "required": ["app", "filter_type"]}},
        {"name": "search_in_favorite_goods", "description": "打开商品收藏夹并按照内容进行搜索 (该函数只涉及淘宝、京东、天猫、闲鱼、阿里巴巴、唯品会、得物)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "search_info": {"type": "string", "description": "搜索的具体内容"}}, "required": ["app", "search_info"]}},
        {"name": "search_in_favorite_stores", "description": "打开店铺收藏夹并按照内容进行搜索 (该函数只涉及抖音、抖音极速版、拼多多、淘宝、京东、阿里巴巴)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "search_info": {"type": "string", "description": "搜索的具体内容"}}, "required": ["app", "search_info"]}},

        # --- 搜索 ---
        {"name": "search_goods", "description": "搜索商品", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "search_info": {"type": "string", "description": "搜索的具体内容"}, "order_type": {"type": "string", "description": "搜索结果的排列方式。默认值: '无'。抖音/抖音极速版/快手/快手极速版/抖音火山版/淘宝/天猫/唯品会支持:['无','综合','销量','价格从低到高','价格从高到低'], 拼多多支持:['无','综合','销量','好评','价格从低到高','价格从高到低'], 京东支持:['无','综合','销量','价格从低到高','价格从高到低','按评论数从高到低'], 闲鱼支持:['无','综合','最近活跃','离我最近','信用','价格从低到高','价格从高到低'], 阿里巴巴支持:['无','综合','销量','价格从低到高','价格从高到低','回头率从高到低'], 得物支持:['无','商品','综合','销量','价格从低到高','价格从高到低'], 转转支持:['无','综合','最新上架','价格从低到高','价格从高到低']"}}, "required": ["app", "search_info", "order_type"]}},
        {"name": "search_stores", "description": "搜索店铺 (该函数不涉及得物、闲鱼、转转、唯品会)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "search_info": {"type": "string", "description": "搜索的具体内容"}, "filter_type": {"type": "string", "description": "对搜索结果进行筛选的条件。默认值: '无'。淘宝支持:['无','好评率100%','好评率99%以上','好评率98%以上','好评率97%以上'], 其他支持的应用均为:['无']"}, "order_type": {"type": "string", "description": "搜索结果的排列方式。默认值: '无'。抖音/抖音极速版支持:['无','综合','销量','人气'], 拼多多支持:['无','综合','销量','评分'], 淘宝支持:['无','综合','销量','信用'], 阿里巴巴支持:['无','综合','销量','回头率'], 其他支持的应用均为:['无']"}}, "required": ["app", "search_info", "filter_type", "order_type"]}},
        {"name": "open_search_history", "description": "打开搜索历史 (该函数涉及全部14个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}},
        {"name": "delete_search_history", "description": "清除搜索历史。危险操作，需进入一级页面后再次确认。(该函数涉及全部14个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}},
        {"name": "open_camera_search", "description": "图片搜索，打开相机功能 (该函数不涉及转转)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}},

        # --- 物流 ---
        {"name": "open_logistics_receive", "description": "打开物流页面查询我购买物品的快递进程", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "filter_type": {"type": "string", "description": "依据物流情况进行筛选的条件。默认值: '无'。抖音/抖音极速版/抖音火山版支持:['无','已签收','待取件','派送中','运送中'], 拼多多支持:['无','我的收件','待发货','运输中','派件中','待取件','已签收','我的寄件','待寄出','已寄出','已取消'], 淘宝支持:['无','取件信息','待取件','派送中','运输中','待发货','已签收'], 京东支持:['无','我收','待揽件','运输中','派送中','已签收','已取消','已拒收','已揽件'], 其他支持的应用均为:['无']"}}, "required": ["app", "filter_type"]}},
        {"name": "open_logistics_send", "description": "打开物流页面查询我寄出物品的物流进程 (该函数不涉及其他10个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "filter_type": {"type": "string", "description": "依据物流情况进行筛选的条件。默认值: '无'。淘宝支持:['无','待寄出寄件','待支付寄件','待取件寄件','派送中寄件','运输中寄件','已签收寄件'], 京东支持:['无','待揽件','运输中','派送中','已签收','已取消','已拒收','已揽件'], 闲鱼/转转支持:['无']"}}, "required": ["app", "filter_type"]}},
        {"name": "open_express_delivery", "description": "打开app的寄快递页面 (该函数只涉及京东)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称，仅限京东"}}, "required": ["app"]}},
        
        # --- 启动 ---
        {"name": "open_app", "description": "打开指定的应用程序 (该函数涉及全部14个app)", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}}
    ]

    all_tools_dict = {tool['name']: tool for tool in all_tools_list}
    
    # --- 2. 加载数据 ---
    # !!! 确保你的CSV文件路径是正确的 !!!
    tool_mapping_path = 'D:/Agent/data/大类-工具映射关系表 - 大类-app-func-new.csv'
    test_cases_path = 'D:/Agent/data/迭代13可用的数据.csv'
    
    tool_mapping_df = load_and_process_tool_mapping(tool_mapping_path)
    try:
        test_cases_df = pd.read_csv(test_cases_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到评测文件: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. 初始化评测结果容器和计数器 ---
    results_list = []
    correct_count = 0
    total_count = len(test_cases_df)
    
    # 创建每日结果文件夹
    output_dir = f"D:/Agent/data/evaluation_results/{time.strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)

    # --- 4. 遍历评测数据集 ---
    for index, row in test_cases_df.iterrows():
        print("\n" + "="*50)
        print(f"正在处理第 {index + 1}/{total_count} 条测试用例...")

        app_name = row['app_name']
        category = row['category']
        user_query = row['query']
        label_str = row['label']

        print(f"  - App: {app_name}")
        print(f"  - Category: {category}")
        print(f"  - Query: {user_query}")

        available_tools = filter_tools(app_name, category, tool_mapping_df, all_tools_dict)

        prediction_str = ""
        if not available_tools:
            print("    ⚠️ 对于此输入，未找到可用工具。")
            prediction_str = json.dumps({"error": "No valid tool or parameter found to fulfill the request"})
        else:
            # 修改了inputSchema，将default字段从里面移除再传给模型，避免混淆
            tools_for_llm = []
            for tool in available_tools:
                clean_tool = json.loads(json.dumps(tool)) # 深拷贝
                for param_name, param_props in clean_tool.get("inputSchema", {}).get("properties", {}).items():
                    if "default" in param_props:
                        del param_props["default"]
                tools_for_llm.append(clean_tool)
            
            input_data = {
                "app": app_name,
                "category": category,
                "user_query": user_query,
                "available_tools": tools_for_llm # 使用清理过的工具列表
            }
            final_input_str = json.dumps(input_data, ensure_ascii=False, indent=2)
            prediction_str = get_qwen_response(instruction, final_input_str, model=model_type)

        eval_result = evaluate_prediction(prediction_str, label_str)
        if eval_result["is_correct"]:
            correct_count += 1
            print("  ✅ 结果正确")
        else:
            print("  ❌ 结果错误")
            print(f"    - 预测: \n{json.dumps(eval_result['prediction'], ensure_ascii=False, indent=2)}")
            print(f"    - 期望: \n{json.dumps(eval_result['label'], ensure_ascii=False, indent=2)}")

        results_list.append({
            'app_name': app_name,
            'category': category,
            'query': user_query,
            'prediction': prediction_str,
            'label': label_str,
            'is_correct': eval_result['is_correct'],
            'tool_name_correct': eval_result['tool_name_correct'],
            'params_correct': eval_result['params_correct']
        })

    # --- 5. 输出评测报告 ---
    print("\n" + "="*50)
    print("评测完成！")
    print(f"\t当前模型（{model_type}）")
    print("="*50)

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"总测试用例: {total_count}")
    print(f"正确数量: {correct_count}")
    print(f"失败数量: {total_count - correct_count}")
    print(f"准确率: {accuracy:.2f}%")

    # --- 6. 保存详细结果到文件 ---
    results_df = pd.DataFrame(results_list)
    output_filename = os.path.join(output_dir, f"evaluation_results_{time.strftime('%H%M%S')}.csv")
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n详细评测结果已保存至: {output_filename}")

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print(f"脚本总执行耗时: {total_duration:.2f} 秒")