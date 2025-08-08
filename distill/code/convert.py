import json
import sys
import re

def convert_to_sharegpt(input_file_path: str, output_file_path: str):
    """
    将专有的 'messages' 格式的 JSONL 文件转换为标准的 ShareGPT 格式。

    转换逻辑:
    - 原始 'user' 内容被拆分为 'system' prompt 和 'human' query。
    - 原始 'assistant' 内容成为 'gpt' 的回复。
    - 忽略原始数据中空的 'system' 消息和其他顶层键。

    参数:
    - input_file_path (str): 输入的 .jsonl 文件路径。
    - output_file_path (str): 输出的 ShareGPT .json 文件路径。
    """
    print(f"--- 开始将 {input_file_path} 转换为 ShareGPT 格式 ---")

    all_conversations = []
    processed_lines = 0
    skipped_lines = 0

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                try:
                    original_data = json.loads(stripped_line)
                    
                    # 验证原始数据格式是否符合预期
                    if 'messages' not in original_data or len(original_data['messages']) < 2:
                        print(f"警告: 第 {line_num} 行缺少 'messages' 字段或内容不足，已跳过。", file=sys.stderr)
                        skipped_lines += 1
                        continue

                    messages = original_data['messages']
                    # 我们主要关心 user 和 assistant 的内容
                    user_message = next((msg for msg in messages if msg.get('role') == 'user'), None)
                    assistant_message = next((msg for msg in messages if msg.get('role') == 'assistant'), None)

                    if not user_message or not assistant_message:
                        print(f"警告: 第 {line_num} 行未找到 'user' 或 'assistant' 角色，已跳过。", file=sys.stderr)
                        skipped_lines += 1
                        continue

                    user_content = user_message['content']
                    assistant_content = assistant_message['content']

                    # 使用正则表达式来稳定地分割 system prompt 和 human query
                    # 这比简单的 split 更健壮，可以处理各种空白符
                    match = re.search(r'\n\n用户的问题或任务是:\s*(.*)', user_content, re.DOTALL)

                    if not match:
                        print(f"警告: 第 {line_num} 行的 'user' 内容格式不正确，找不到任务描述，已跳过。", file=sys.stderr)
                        skipped_lines += 1
                        continue
                    
                    # 分割点之前是 system prompt
                    system_prompt = user_content[:match.start()].strip()
                    # 分割点之后是 human query
                    human_query = match.group(1).strip()
                    
                    # 构建 ShareGPT 格式的对话轮次
                    sharegpt_turns = [
                        {
                            "from": "system",
                            "value": system_prompt
                        },
                        {
                            "from": "human",
                            "value": human_query
                        },
                        {
                            "from": "gpt",
                            "value": assistant_content
                        }
                    ]
                    
                    # 将这组对话添加到总列表中
                    # ShareGPT 的标准格式是一个包含 "conversations" 键的字典
                    all_conversations.append({
                        "conversations": sharegpt_turns
                    })
                    processed_lines += 1

                except (json.JSONDecodeError, TypeError) as e:
                    print(f"警告: 第 {line_num} 行解析失败，错误: {e}。已跳过。", file=sys.stderr)
                    skipped_lines += 1

        # 将所有转换后的对话写入一个新的JSON文件
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # indent=2 是 ShareGPT 格式常用的缩进
            json.dump(all_conversations, outfile, indent=2, ensure_ascii=False)

        print("\n--- 转换成功！ ---")
        print(f"  成功处理了 {processed_lines} 条对话。")
        if skipped_lines > 0:
            print(f"  跳过了 {skipped_lines} 条格式不兼容的记录。")
        print(f"  ShareGPT 格式的数据已保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"\n错误: 输入文件 '{input_file_path}' 未找到。", file=sys.stderr)
    except Exception as e:
        print(f"\n在处理过程中发生未知错误: {e}", file=sys.stderr)

# ==============================================================================
#  使用方法:
#  1. 将下面的文件名替换成你的实际文件。
#  2. 在终端中运行 `python convert_to_sharegpt.py`。
# ==============================================================================
if __name__ == '__main__':
    # --- 请在这里配置您的文件路径 ---
    input_path = '/home/workspace/LLaMA-Factory/data/train_迭代5forlevel0badcase0801_merge.jsonl'
    output_path = '/home/workspace/LLaMA-Factory/data/train_迭代5forlevel0badcase0801_merge_sharegpt_formatted_data.json'
    # --------------------------------

    # 调用转换函数
    convert_to_sharegpt(input_path, output_path)