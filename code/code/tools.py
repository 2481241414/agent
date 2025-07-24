from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from loguru import logger

# 初始化 FastMCP 服务器
mcp = FastMCP("tools")
logger.debug("FastMCP 服务器启动中...")

@mcp.tool()
async def count_letters(word: str, letter: str) -> str:
    """统计单词中特定字母出现的次数。

    参数:
        word: 要分析的单词
        letter: 要统计的字母（单个字符）
    """
    # 确保 letter 是单个字符
    if len(letter) != 1:
        return "请提供单个字母进行统计"
    
    # 将输入转换为小写以忽略大小写差异
    word = word.lower()
    letter = letter.lower()
    
    # 统计字母出现次数
    count = word.count(letter)
    
    return f"单词 '{word}' 中有 {count} 个 '{letter}'"

@mcp.tool()
async def compare_expressions(expr1: str, expr2: str) -> str:
    """比较两个数字或数学表达式的大小。
    
    支持数字的四则运算(+,-,*,/)，可以比较类似"99-1"和"98"这样的表达式大小。
    
    参数:
        expr1: 第一个表达式，如 "99", "2*3+4"
        expr2: 第二个表达式，如 "98", "10*2"
    """
    try:
        # 计算第一个表达式的值
        value1 = eval(expr1)
        
        # 计算第二个表达式的值
        value2 = eval(expr2)
        
        # 比较结果
        if value1 > value2:
            return f"表达式 \"{expr1}\" (计算结果: {value1}) 比 \"{expr2}\" (计算结果: {value2}) 大"
        elif value1 < value2:
            return f"表达式 \"{expr1}\" (计算结果: {value1}) 比 \"{expr2}\" (计算结果: {value2}) 小"
        else:
            return f"表达式 \"{expr1}\" 和 \"{expr2}\" 相等 (都等于 {value1})"
    
    except Exception as e:
        # 处理计算错误
        return f"计算错误: {str(e)}。请确保输入的是有效的数学表达式。"
    
@mcp.tool()
async def get_current_date_time() -> str:
    """获取当前时间和日期。"""
    import datetime
    now = datetime.datetime.now()
    return f'当前时间为：{now.strftime("%Y-%m-%d %H:%M:%S")}'

@mcp.tool()
async def save_to_file(content) -> str:
    """将内容保存到文件中。

    参数:
        content: 要保存的内容
    """
    try:
        with open('data.txt', 'w') as f:
            f.write(content)
        return f"内容已保存到文件 '{'data.txt'}'"
    except Exception as e:
        return f"保存文件时出现错误: {e}"

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='stdio')