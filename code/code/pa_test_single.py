from openai import OpenAI
import json
import time
import os
import sys

# ... (你的 client 初始化代码保持不变) ...
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key is None:
    print("警告：未在环境变量中找到 DASHSCOPE_API_KEY，将使用代码中硬编码的Key。", file=sys.stderr)
    api_key = "sk-4fcc85e2509649198bdcafa4e985ce6e" 

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def decompose_query(user_query):
    # 【已升级】这是支持多任务拆解的Prompt模板
    prompt_template = """
# 角色
你是一个顶级的多任务拆解专家。你的专长是将用户可能包含多个意图的复杂、口语化的查询，分解成一个或多个清晰、独立、可执行的子任务列表。

# 任务
你的任务是根据用户输入的查询（query），将其拆解成一个JSON格式的子任务列表。每个子任务都必须是独立的，并且能够被后续程序直接执行。

# 输出规范
1.  最终输出必须是一个JSON数组（列表）。
2.  如果原始query只包含一个任务，则列表中只有一个JSON对象。
3.  如果原始query包含多个任务，则列表中有多个JSON对象。
4.  如果原始query不包含任何可识别的任务，则返回一个空列表 `[]`。
5.  列表中的每个JSON对象都必须包含以下字段：
    *   `app_name`: 必须从列表中选择：[抖音, 抖音极速版, 快手, 快手极速版, 拼多多, 淘宝, 京东, 天猫, 闲鱼, 抖音火山版, 阿里巴巴, 唯品会, 得物, 转转]。注意识别别名。
    *   `category`: 必须从列表中选择：[签到, 收藏, 搜索, 物流, 订单, 发票, 购物车, 客服, 启动]。
    *   `decomposed_query`: 重新生成的、清晰的、针对单个任务的指令性查询。

# 思考过程
请遵循以下思考过程来完成任务：
1.  仔细阅读用户查询，判断它包含一个还是多个潜在的任务点（例如，多个动作或多个对象）。
2.  识别查询中的共享上下文信息，例如App名称，它可能适用于所有子任务。
3.  对于每一个识别出的任务点：
    a. 确定其`category`（意图）。
    b. 确定其`app_name`（实体），应用共享上下文（如果存在）。
    c. 提取任务的关键参数（例如搜索的物品、查看的订单类型等）。
    d. 基于以上信息，生成一句清晰、独立的`decomposed_query`。
4.  将所有拆解出的子任务对象组合成一个JSON列表。

# 示例
---
query: "用淘宝查看火车和机票"
JSON输出:
[
  {{
    "app_name": "淘宝",
    "category": "搜索",
    "decomposed_query": "在淘宝中搜索火车票"
  }},
  {{
    "app_name": "淘宝",
    "category": "搜索",
    "decomposed_query": "在淘宝中搜索机票"
  }}
]
---
query: "帮我在京东看一下购物车，顺便查下昨天的订单物流"
JSON输出:
[
  {{
    "app_name": "京东",
    "category": "购物车",
    "decomposed_query": "在京东中查看购物车"
  }},
  {{
    "app_name": "京东",
    "category": "物流",
    "decomposed_query": "在京东中查看昨天的订单物流"
  }}
]
---
query: "快告诉我抖音极速版签到领金币的任务在哪！"
JSON输出:
[
  {{
    "app_name": "抖音极速版",
    "category": "签到",
    "decomposed_query": "在抖音极速版中打开签到领金币任务"
  }}
]
---
query: "帮我打开抖音，然后在淘宝搜一下新出的手机壳"
JSON输出:
[
  {{
    "app_name": "抖音",
    "category": "启动",
    "decomposed_query": "打开抖音"
  }},
  {{
    "app_name": "淘宝",
    "category": "搜索",
    "decomposed_query": "在淘宝中搜索新出的手机壳"
  }}
]
---

# 开始任务
现在，请根据以上规范，处理以下新的用户查询。

query: "{query}"
"""
    
    # 替换查询（注意，这次我们不需要修改大括号，因为JSON示例被包裹在markdown中，或者我们可以用f-string）
    full_prompt = prompt_template.format(query=user_query)
    # print(full_prompt) # 调试时可以取消注释

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": "你将严格按照指令进行多任务拆解，并只输出JSON格式的列表。不要包含任何解释或代码块标记。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
        )
        end_time = time.time()
        latency = end_time - start_time
        print(f"    API 调用耗时: {latency:.4f} 秒")
        
        result_str = response.choices[0].message.content
        
        # 健壮性处理
        if result_str.strip().startswith("```json"):
            result_str = result_str.strip()[7:-3].strip()
            
        # 解析JSON列表
        result_list = json.loads(result_str)
        return result_list

    except json.JSONDecodeError as e:
        print(f"JSON解码失败: {e}", file=sys.stderr)
        print(f"模型返回的原始字符串: '{result_str}'", file=sys.stderr)
        return [{"error": "JSON Error", "message": str(e)}]
    except Exception as e:
        print(f"发生未知错误: {e}", file=sys.stderr)
        return [{"error": "Unknown Error", "message": str(e)}]

# --- 测试 ---
print("--- 测试多任务查询 ---")
multi_task_query = "用淘宝查看火车和机票，然后帮我在pdd上看看购物车里有啥。"
decomposed_tasks = decompose_query(multi_task_query)

print(f"原始Query: {multi_task_query}")
print("拆解后的任务列表:")
print(json.dumps(decomposed_tasks, indent=2, ensure_ascii=False))

print("\n--- 测试单任务查询 ---")
single_task_query = "我想在1688上看看我的购物车"
decomposed_single_task = decompose_query(single_task_query)

print(f"原始Query: {single_task_query}")
print("拆解后的任务列表:")
print(json.dumps(decomposed_single_task, indent=2, ensure_ascii=False))