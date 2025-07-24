import os
import sys
import json
import time
import re
import pandas as pd # 引入 pandas 库
from openai import OpenAI

# --- 最佳实践：从环境变量中读取API Key ---
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key is None:
    print("警告：未在环境变量中找到 DASHSCOPE_API_KEY，将使用代码中硬编码的Key。", file=sys.stderr)
    api_key = "sk-4fcc85e2509649198bdcafa4e985ce6e" # 替换为你的Key

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_qwen_response(instruction: str, user_input: str, model: str = "qwen2.5-7b-instruct") -> str:
    """调用通义千问模型获取响应的函数。"""
    try:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
        ]
        
        print(f"--- 正在向模型 '{model}' 发送请求 ---")
        # 为了简洁，可以注释掉下面两行超长内容的打印
        # print(f"System: {instruction}")
        # print(f"User: {user_input}")
        
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"enable_thinking": False},
        )
        end_time = time.time()
        latency = end_time - start_time
        
        response_content = completion.choices[0].message.content
        print("--- 模型响应 ---")
        print(f"API 调用耗时: {latency:.4f} 秒")
        return response_content
        
    except Exception as e:
        error_message = f"调用API时出错: {e}"
        print(error_message, file=sys.stderr)
        return error_message

# --- 新增的数据处理和工具过滤逻辑 ---

def load_and_process_tool_mapping(filepath: str) -> pd.DataFrame:
    """
    加载并处理CSV格式的工具映射数据（为每行都完整的格式优化）。
    
    Args:
        filepath (str): CSV文件的路径。

    Returns:
        pd.DataFrame: 包含 '大类', 'app', 'tool_name' 列的DataFrame。
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        # 核心修正：不再使用 fillna，因为新数据格式每行都是完整的。
        
        # 清理可能存在的前后空格，并提取工具名
        # 例如从 "  search_goods(app, ...)" 中提取 "search_goods"
        df['大类'] = df['大类'].str.strip()
        df['app'] = df['app'].str.strip()
        df['tool_name'] = df['function_name'].str.strip().str.extract(r'(\w+)', expand=False)
        
        # 去除因为任何原因导致无法提取tool_name的行
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
    """
    根据app和category筛选可用的工具列表。

    Args:
        app_name (str): 当前的应用名称。
        category (str): 当前的大类名称。
        mapping_df (pd.DataFrame): 预处理过的工具映射DataFrame。
        all_tools_dict (dict): 包含所有工具完整定义的字典。

    Returns:
        list: 筛选后得到的工具定义列表。
    """
    print(f"\n--- 正在根据 app='{app_name}' 和 category='{category}' 进行工具过滤 ---")
    
    # 筛选出符合 app 和 大类 的行
    filtered_df = mapping_df[
        (mapping_df['app'] == app_name) & 
        (mapping_df['大类'] == category)
    ]
    
    # 获取唯一的工具名称列表
    tool_names = filtered_df['tool_name'].unique().tolist()
    # print(tool_names)
    
    # 根据工具名称从总工具字典中查找完整的工具定义
    available_tools = [all_tools_dict[name] for name in tool_names if name in all_tools_dict]
    # print(available_tools)
    
    print(f"🔍 过滤后的可用工具: {[tool['name'] for tool in available_tools]}")
    return available_tools


# --- 主程序 ---

if __name__ == "__main__":
    overall_start_time = time.time()

    # 优化的Instruction Prompt (无需改动)
    instruction = """
# 角色
你是一个AI工具调用专家。你的任务是根据用户输入，选择最合适的工具并填充参数，然后以JSON格式输出。

# 输入格式
你会收到一个JSON对象，包含四个键：
1. `app`: (string) 应用名称。
2. `category`: (string) 功能分类，用于快速筛选工具。
3. `user_query`: (string) 用户的原始自然语言请求。
4. `available_tools`: (array) 本次任务可用的工具列表。

# 核心指令：你必须严格遵循以下三步流程

1.  **理解意图 (Understand Intent)**
    *   **优先使用 `category` 进行初步筛选**。`category` 是最关键的线索。例如，`category`为"订单"时，应重点关注与订单相关的工具。
    *   结合 `user_query` 理解用户的具体动作和需求。例如，在“订单”类别下，“我的待付款订单”表示查看，而“搜一下我买的手机”表示搜索。

2.  **选择工具 (Select Tool)**
    *   在 `available_tools` 列表中，根据上一步的综合理解，选择一个最匹配的工具。请仔细比对工具的 `name` 和 `description` 来做最终决定。

3.  **填充参数 (Fill Parameters)**
    *   仔细阅读所选工具的 `inputSchema`。
    *   **`app` 参数**：【必须】直接使用输入中提供的 `app` 值，【绝不能】从 `user_query` 中提取。
    *   **其他参数**：从 `user_query` 中提取信息进行填充。
    *   对于可选参数，如果 `user_query` 未提供信息，且其描述中指明可以为 `null`，则赋值 `null`。否则，在输出中忽略该参数。

# 输出格式
*   最终输出【必须】是一个单一、严格的JSON对象。
*   【禁止】包含任何解释、注释或Markdown标记 (如 ```json ... ```)。
*   JSON结构必须为: `{ "tool_name": "...", "tool_parameters": { ... } }`

# 示例 (学习这个思考过程和输出)

---
### 示例输入:
```json
{
  "app": "淘宝",
  "category": "订单",
  "user_query": "帮我看看待发货的订单",
  "available_tools": [
    {
      "name": "view_orders",
      "description": "查看指定状态的订单列表",
      "inputSchema": {
        "type": "object",
        "properties": {
          "app": {"type": "string"},
          "order_status": {"type": "string", "description": "订单状态, 如待发货、待付款等; 若未提及则为null"}
        },
        "required": ["app", "order_status"]
      }
    },
    {
      "name": "search_order",
      "description": "根据关键词搜索订单",
      "inputSchema": {
        "type": "object",
        "properties": {
          "app": {"type": "string"},
          "search_info": {"type": "string"}
        },
        "required": ["app", "search_info"]
      }
    }
  ]
}
```

### 思考过程:
1.  **分析输入**: `app`是"淘宝", `category`是"订单", `user_query`是"帮我看看待发货的订单"。
2.  **筛选工具**: `category`是"订单"，所以我优先考虑和订单相关的工具。`view_orders`和`search_order`都相关。
3.  **确定动作**: `user_query`是查看"待发货"的订单，这明确匹配`view_orders`工具，而不是搜索。
4.  **填充参数**:
    *   `app`: 直接使用输入 "淘宝"。
    *   `order_status`: 从 "待发货的订单" 中提取出 "待发货"。
5.  **构建输出**: 生成最终的JSON。

### 示例输出:
```json
{
  "tool_name": "view_orders",
  "tool_parameters": {
    "app": "淘宝",
    "order_status": "待发货"
  }
}
```
"""

    # 1. 定义所有可能的工具的完整Schema
    # 1. 定义所有可能的工具的完整Schema (根据文档内容已补全)
    all_tools_list = [
        # --- 订单 (Order) ---
        {
            "name": "view_orders",
            "description": "在app应用程序中查看指定状态的订单列表,例如待付款、待收货、待评价等。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "order_status": {
                        "type": "string",
                        "description": "订单状态。如果用户无提及,传null。各App支持的值如下：阿里巴巴:['待付款', '待发货', '待收货', '待评价', '退款-售后'], 淘宝:['待付款', '待发货', '待收货', '待评价', '退款/售后'], 天猫:['待付款', '待发货', '待收货', '待评价', '退款和售后'], 抖音/抖音火山版/抖音极速版/快手/快手极速版:['待支付', '待发货', '待收货', '待评价', '售后'], 得物:['待付款', '待发货', '待收货'], 京东:['待付款', '待收货', '待服务', '待评价', '退款/售后'], 拼多多:['待付款', '待收货', '待分享', '待发货', '退款售后'], 唯品会:['已取消的待付款', '已完成的待付款', '待付款', '待评价', '待收货', '退换售后']"
                    }
                },
                "required": ["app"] # order_status 可为 null，所以从 required 中移除
            }
        },
        {
            "name": "view_second_hand_orders",
            "description": "在二手交易平台中查看指定状态的订单列表或商品。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "order_status": {
                        "type": "string",
                        "description": "订单状态。如果用户无提及,传null。各App支持的值如下：闲鱼:['全部', '待付款', '待发货', '待收货', '待评价', '退款中'], 转转:['待付款', '待发货', '待收货', '待评价', '退款售后', '未完结的退款售后', '待评价的退款售后']"
                    },
                    "product_status": {
                        "type": "string",
                        "description": "用户售卖商品的状态。各App支持的值如下：闲鱼:['在卖', '草稿', '已下架'], 转转:['进行中的', '已完成的', '已关闭的']"
                    },
                    "transaction_type": {
                        "type": "string",
                        "description": "区分用户的状态是买入还是卖出。各App支持的值如下：闲鱼:['我买到的', '我发布的', '我卖出的'], 转转:['我发布的', '我卖的', '我购买的']"
                    }
                },
                "required": ["app"]
            }
        },
        {
            "name": "search_order",
            "description": "在app应用程序中根据关键词搜索订单。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "应用名称"},
                    "search_info": {"type": "string", "description": "搜索内容"},
                    "order_status": {
                        "type": "string",
                        "description": "对订单的限定条件。各App支持的值如下：唯品会:['退换售后'], 闲鱼:['我发布的', '我卖出的', '我买到的'], 转转:['我购买的', '我卖出的']"
                    }
                },
                "required": ["app", "search_info"]
            }
        },
        {"name": "view_reviews", "description": "在app应用程序中查看我的评价", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}},
        {"name": "apply_after_sales", "description": "在app应用程序中申请退款或售后", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}},

        # --- 发票 (Invoice) ---
        {
            "name": "open_invoice_page",
            "description": "在app应用程序中打开与发票相关的页面。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "应用名称"},
                    "page_type": {
                        "type": "string",
                        "description": "打开的页面类型。各App支持的值如下：得物:['发票服务'], 京东:['发票抬头管理'], 淘宝:['发票中心'], 唯品会:['发票服务']"
                    }
                },
                "required": ["app", "page_type"]
            }
        },
        {
            "name": "view_invoice_info",
            "description": "在app应用程序中查看指定状态的发票相关信息。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "应用名称"},
                    "invoice_status": {
                        "type": "string",
                        "description": "发票的状态。如果是全部发票或未提及,则为null。各App支持的值如下：京东:['全部', '换开/合开', '专票提交', '申请记录'], 拼多多:['全部', '已开票', '申请中', '待申请'], 淘宝:['申请中', '已开票', '未申请'], 唯品会:['开具发票', '我的发票']"
                    }
                },
                "required": ["app"]
            }
        },

        # --- 购物车 (Shopping Cart) ---
        {
            "name": "view_cart_content",
            "description": "在app应用程序中查看购物车/采购车(阿里巴巴的叫法)中指定类型的商品。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "应用名称"},
                    "filter_type": {
                        "type": "string",
                        "description": "指定的商品类型。如果是全部或未提及,则为null。各App支持的值如下：阿里巴巴:['常购', '代发', '现货'], 得物:['降价'], 抖音:['有货', '降价'], 抖音火山版:['降价'], 抖音极速版:['有货', '降价'], 京东:['送礼', '凑单', '跨店满减', '降价'], 淘宝:['降价', '常购', '失效'], 唯品会:['近期加购', '降价', '有货']"
                    }
                },
                "required": ["app"]
            }
        },
        {"name": "search_cart_content", "description": "在app应用程序的购物车/采购车(阿里巴巴的叫法)中查找商品", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}, "search_info": {"type": "string", "description": "搜索内容"}}, "required": ["app", "search_info"]}},
        {
            "name": "open_cart_page",
            "description": "在app应用程序中打开购物车的功能界面，目前主要针对淘宝。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "应用名称"},
                    "page_type": {
                        "type": "string",
                        "description": "购物车中的功能界面。淘宝支持:['管理', '分组']"
                    }
                },
                "required": ["app", "page_type"]
            }
        },

        # --- 客服 (Customer Service) ---
        {"name": "open_customer_service", "description": "在app应用程序中联系官方客服", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "应用名称"}}, "required": ["app"]}},

        # --- 签到 (Sign In) ---
        {
            "name": "sign_in",
            "description": "在app程序中完成每日签到,领取积分、金币等奖励的操作。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "page": {
                        "type": "string",
                        "description": "用于区分同一app的不同签到功能,需要给出具体子页面的名称。各App支持的值如下：抖音极速版:['赚钱', '我的钱包'], 淘宝:['淘金币', '红包'], 天猫:['领现金', '红包']"
                    }
                },
                "required": ["app", "page"] # page是必须的，因为它区分了具体功能
            }
        },

        # --- 收藏 (Favorites) ---
        {
            "name": "open_favorite_goods",
            "description": "在app程序中打开收藏的喜爱、想要或关注商品的页面,并可按照条件进行筛选和排序。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "filter": {
                        "type": "string",
                        "description": "在商品收藏夹中具体应用的筛选条件。各App支持的值如下：淘宝/天猫:['有降价', '已买过', '低库存', '已失效', '7天内', '30天内', '90天内', '半年前', '一年前'], 京东:['降价', '促销', '有货', '下架'], 闲鱼:['降价', '有效', '失效'], 阿里巴巴:['降价', '低价补货', '买过'], 唯品会:['降价', '有货', '已失效', '7天内', '1个月内', '3个月内', '半年前']"
                    },
                    "order": {
                        "type": "string",
                        "description": "查看商品收藏夹时使用的商品排列方式。各App支持的值如下：淘宝/天猫:['最近收藏在前', '最早收藏在前']"
                    }
                },
                "required": ["app"]
            }
        },
        {
            "name": "open_favorite_stores",
            "description": "在app程序中打开收藏的喜爱或关注店铺的页面,并可按照条件进行筛选。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "filter": {
                        "type": "string",
                        "description": "在店铺收藏夹中具体应用的筛选条件。淘宝支持:['特别关注', '直播中', '有上新']"
                    }
                },
                "required": ["app"]
            }
        },
        {"name": "search_in_favorite_goods", "description": "在app程序中打开收藏商品页面,并在其中的搜索栏中进行搜索。", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "app应用程序的名称"}, "search_info": {"type": "string", "description": "搜索的具体内容"}}, "required": ["app", "search_info"]}},
        {"name": "search_in_favorite_stores", "description": "在app程序中打开收藏店铺页面,并在其中的搜索栏搜索商品。", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "app应用程序的名称"}, "search_info": {"type": "string", "description": "搜索的具体内容"}}, "required": ["app", "search_info"]}},

        # --- 搜索 (Search) ---
        {
            "name": "search_goods",
            "description": "在app程序中依据名称搜索商品,可以指定搜索结果的排序方式。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "search_info": {"type": "string", "description": "搜索的具体内容"},
                    "order": {
                        "type": "string",
                        "description": "搜索结果的排列方式。各App支持的值如下：抖音/抖音极速版/快手/快手极速版/抖音火山版/淘宝/天猫/唯品会/得物:['综合', '销量', '价格从低到高', '价格从高到低'], 拼多多:['综合', '销量', '好评', '价格从低到高', '价格从高到低'], 京东:['综合', '销量', '价格从低到高', '价格从高到低', '按评论数从高到低'], 闲鱼:['综合', '最近活跃', '离我最近', '信用', '价格从低到高', '价格从高到低'], 阿里巴巴:['综合', '销量', '价格从低到高', '价格从高到低', '回头率从高到低'], 转转:['综合', '最新上架', '价格从低到高', '价格从高到低']"
                    }
                },
                "required": ["app", "search_info"]
            }
        },
        {
            "name": "search_stores",
            "description": "在app程序中依据名称搜索店铺,可以使用筛选器限制搜索结果,也可以指定搜索结果的排序方式。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "search_info": {"type": "string", "description": "搜索的具体内容"},
                    "filter": {
                        "type": "string",
                        "description": "对搜索结果进行筛选的条件。淘宝支持:['好评率100%', '好评率99%以上', '好评率98%以上', '好评率97%以上']"
                    },
                    "order": {
                        "type": "string",
                        "description": "搜索结果的排列方式。各App支持的值如下：抖音/抖音极速版:['综合', '销量', '人气'], 拼多多:['综合', '销量', '评分'], 淘宝:['综合', '销量', '信用'], 阿里巴巴:['综合', '销量', '回头率']"
                    }
                },
                "required": ["app", "search_info"]
            }
        },
        {"name": "open_search_history", "description": "打开app程序的搜索历史界面。", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "app应用程序的名称"}}, "required": ["app"]}},
        {"name": "delete_search_history", "description": "清除app中的搜索历史。这是一个危险操作。", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "app应用程序的名称"}}, "required": ["app"]}},
        {"name": "open_camera_search", "description": "打开app程序的图片搜索(拍照/扫一扫)功能。", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "app应用程序的名称"}}, "required": ["app"]}},

        # --- 物流 (Logistics) ---
        {
            "name": "open_order_interface",
            "description": "打开显示已购或已售商品信息的订单界面,查看相关订单信息,并根据物流情况进行筛选。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "app应用程序的名称"},
                    "filter": {
                        "type": "string",
                        "description": "在订单界面依据物流情况进行筛选的筛选条件。各App支持的值如下：抖音/抖音极速版/抖音火山版:['已签收', '待取件', '派送中', '运送中'], 拼多多:['我的收件', '待发货', '运输中', '派件中', '待取件', '已签收', '我的寄件', '待寄出', '已寄出', '已取消'], 淘宝:['取件', '待取件', '派送中', '运输中', '待发货', '已签收', '寄件', '待寄出寄件', '待支付寄件', '待取件寄件', '派送中寄件', '运输中寄件', '已签收寄件'], 京东:[针对'我收'的物流状态:'我收', '待揽件我收', '运输中我收', '派送中我收', '已签收我收', '已取消我收', '已拒收我收', '已揽件我收'; 针对'我寄'的物流状态:'我寄', '待揽件我寄', '运输中我寄', '派送中我寄', '已签收我寄', '已取消我寄', '已拒收我寄', '已揽件我寄'], 闲鱼:['卖出', '买到'], 转转:['我买(默认)', '我卖']"
                    }
                },
                "required": ["app"]
            }
        },
        {"name": "use_express_delivery", "description": "打开app寄送快递的界面。", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "app应用程序的名称"}}, "required": ["app"]}},

        # --- 启动 (Launch) ---
        {"name": "open_app", "description": "打开指定的应用程序。", "inputSchema": {"type": "object", "properties": {"app": {"type": "string", "description": "app应用程序的名称"}}, "required": ["app"]}}
    ]
    # 转换为字典以加快查找速度
    all_tools_dict = {tool['name']: tool for tool in all_tools_list}

    # 2. 加载并处理CSV映射文件
    tool_mapping_df = load_and_process_tool_mapping('D:\Agent\data\大类-工具映射关系表 - 大类-app-func.csv')

    # 3. 定义你的测试用例
    # user_query = "我想看看我的淘宝待付款订单"
    # app_name = "淘宝"
    # category = "订单"

    # user_query = "在快手极速版中搜索哪吒2"
    # app_name = "快手极速版"
    # category = "搜索" 

    
    # user_query = "在淘宝的收藏夹里找一下我关注的店铺"
    # app_name = "淘宝"
    # category = "收藏"
    
    user_query = "去淘宝找客服聊聊"
    app_name = "淘宝" 
    category = "客服"

    # 4. 【核心逻辑】根据输入进行工具过滤
    available_tools = filter_tools(app_name, category, tool_mapping_df, all_tools_dict)
    
    # 5. 如果没有可用的工具，则不调用LLM，直接退出
    if not available_tools:
        print(f"\n错误：对于应用 '{app_name}' 和大类 '{category}'，没有找到可用的工具。程序终止。")
    else:
        # 6. 组合成最终的输入并调用LLM
        input_data = {
            "app": app_name,
            "category": category,
            "user_query": user_query,
            "available_tools": available_tools # 使用过滤后的工具列表
        }

        final_input_str = json.dumps(input_data, ensure_ascii=False, indent=2)
        result = get_qwen_response(instruction, final_input_str)

        # 7. 打印结果
        print("\n--- 最终输出 (JSON) ---")
        try:
            pretty_result = json.dumps(json.loads(result), ensure_ascii=False, indent=2)
            print(pretty_result)
        except json.JSONDecodeError:
            print(result) # 如果返回的不是合法的JSON，则直接打印

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print("\n" + "-" * 20)
    print(f"脚本总执行耗时: {total_duration:.4f} 秒")