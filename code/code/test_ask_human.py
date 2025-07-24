from openai import OpenAI


tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_human",
            "description": "使用这个工具向用户发起提问用来补充信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "inquire": {
                        "type": "string",
                        "description": "想要向用户提问的问题",
                    }
                },
                "required": ["inquire"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_order",
            "description": "在app应用程序中搜索指定内容的订单",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {
                        "type": "string",
                        "description": "应用名称"
                    },
                    "search_info": {
                        "type": "string",
                        "description": "用户输入的搜索订单内容，值不能为空"
                    },
                    "order_status": {
                        "type": "string",
                        "description": "对订单的限定条件，例如退款售后、我发布的、我卖出的等",
                    }
                },
                "required": ["search_info"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "order_taxi",
            "description": "用于打车的工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "目的地" 
                    },
                    "departure": {
                        "type": "string",
                        "description": "出发地"
                    }
                },
                "required": ["destination", "departure"],
            },
        }
    }
]


class QWENMODELS:
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = "sk-4fcc85e2509649198bdcafa4e985ce6e"
    model_list = [
        'qwen-plus',
        "qwen-turbo",
        "qwen2.5-7b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-3b-instruct"
    ]


client = OpenAI(
    api_key=QWENMODELS.api_key,
    base_url=QWENMODELS.base_url,
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个具有工具选择能力的智能体"},
        {"role": "user", "content": "帮我打个到北京的车"},
    ],
    tools=tools
)
print(completion)
