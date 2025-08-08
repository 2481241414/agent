import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os
import json
from loguru import logger

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

# 从 .env 文件中加载环境变量
load_dotenv()

# 获取必要的环境变量
api_key = os.environ["DS_API_KEY"]
base_url = os.environ["DS_API_BASE"]
model_name = os.environ["API_MODEL_NAME"]
max_tool_calls_allowed = 5  # 每轮对话中允许最多调用 5 次工具

logger.debug("FastMCP 服务器启动中...")

class MCPClient:
    def __init__(self):
        # 初始化 session 和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # 初始化 OpenAI 客户端
        self.openai = OpenAI(api_key=api_key, base_url=base_url)

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器

        参数:
            server_script_path: 服务器脚本路径 (.py 或 .js 文件)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        # 启动子进程并建立标准输入输出通信
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # 初始化客户端 session
        await self.session.initialize()

        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print(
            "\n成功连接到服务器，检测到的工具：",
            [[tool.name, tool.description, tool.inputSchema] for tool in tools],
        )

    async def process_query(self, query: str) -> str:
        """处理用户查询，支持多轮工具调用"""
        messages = [{"role": "user", "content": query}]

        # 获取当前可用工具
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": getattr(tool, "inputSchema", {}),
                },
            }
            for tool in response.tools
        ]

        # 开始对话循环
        current_tool_calls_count = 0
        while True:  # 不限制总调用次数，但会受到 max_tool_calls_allowed 控制
            # 调用模型生成回复
            model_response = self.openai.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=available_tools,
                max_tokens=1000
            )

            assistant_message = model_response.choices[0].message
            logger.debug(f"助手返回消息: {assistant_message}")

            # 将助手回复加入对话消息中
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": getattr(assistant_message, "tool_calls", None)
            })

            # 判断是否需要调用工具
            if not hasattr(assistant_message, "tool_calls") or not assistant_message.tool_calls or max_tool_calls_allowed <= current_tool_calls_count:
                # 无需调用工具，直接返回最终回复
                return assistant_message.content or ""

            # 当前轮处理所有工具调用
            for tool_call in assistant_message.tool_calls:
                try:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.debug(f"执行工具: {tool_name}，参数: {tool_args}")
                    result = await self.session.call_tool(tool_name, tool_args)
                    logger.debug(f"工具返回结果: {result}")

                    # 保证结果是字符串
                    if isinstance(result, bytes):
                        result = result.decode('utf-8', errors='replace')
                    elif not isinstance(result, str):
                        result = str(result)

                    # 工具调用结果加入对话
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })

                except Exception as e:
                    error_msg = f"工具调用失败: {str(e)}"
                    logger.error(error_msg)
                    messages.append({
                        "role": "tool",
                        "content": f"Error: {str(e)}",
                        "tool_call_id": tool_call.id
                    })
            
            current_tool_calls_count += 1
            if current_tool_calls_count >= max_tool_calls_allowed:
                logger.warning("工具调用次数过多，停止调用。")
            
            # 循环继续，根据模型判断是否继续调用工具

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动！")
        print("输入你的问题，或输入 'quit' 退出。")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                logger.exception("聊天循环中出错")
                print(f"\n出错了: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("用法: python client.py <服务器脚本路径>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
