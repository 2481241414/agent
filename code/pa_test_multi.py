import pandas as pd
import requests # <--- 引入 requests 库
import json
import time
import os
import sys
import io
from tqdm import tqdm
from datetime import datetime # 【新增】引入datetime库用于生成时间戳

# --- 1. 移除 OpenAI Client，定义请求参数 ---
# (此部分无需改动)
# 本地模型服务的URL和头部信息
URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    # 如果你的本地服务需要授权，请取消下面的注释并填入正确的token
    # "Authorization": "Bearer YOUR_LOCAL_API_KEY"
}
MODEL_NAME = "qwen2.5-7b-instruct" # <-- 重要：请确保这与你本地加载的模型名称一致

# --- 2. 使用 requests.post 重写 decompose_query 函数 ---
# (此部分无需改动)
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
    
    full_prompt = prompt_template.format(query=user_query)

    # 构造请求体 (payload)
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你将严格按照指令进行多任务拆解，并只输出JSON格式的列表。不要包含任何解释或代码块标记。"},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0
    }

    try:
        # 直接使用 requests.post 发送请求
        response = requests.post(URL, headers=HEADERS, json=data, timeout=60)
        response.raise_for_status()  # 如果HTTP状态码是4xx或5xx，则抛出异常
        
        resp_json = response.json()
        result_str = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')

        if not result_str:
            return [{"error": "API Error", "message": "模型返回了空内容"}]
        
        if result_str.strip().startswith("```json"):
            result_str = result_str.strip()[7:-3].strip()
            
        result_list = json.loads(result_str)
        return result_list

    except requests.exceptions.RequestException as e:
        print(f"\n网络请求失败: {e}", file=sys.stderr)
        return [{"error": "Network Error", "message": str(e)}]
    except json.JSONDecodeError as e:
        print(f"\nJSON解码失败: {e}", file=sys.stderr)
        print(f"模型返回的原始字符串: '{result_str}'", file=sys.stderr)
        return [{"error": "JSON Error", "message": str(e)}]
    except Exception as e:
        print(f"\n发生未知错误: {e}", file=sys.stderr)
        return [{"error": "Unknown Error", "message": str(e)}]


# --- 3. 批量测试与评估主程序 【修改】---
def batch_test_and_evaluate(dataset_path):
    """
    读取CSV数据集，进行批量测试，实时打印每个样本的结果，并在最后计算和保存评估结果。
    """
    print(f"正在从 {dataset_path} 加载数据集...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"错误：找不到数据集文件 {dataset_path}", file=sys.stderr)
        return

    # 找到第一个非空的列名，并以此为基准重命名
    first_valid_col = df.columns[0]
    col_map = {
        first_valid_col: 'category_label',
        df.columns[1]: 'app_name_label',
        df.columns[3]: 'final_query',
        df.columns[4]: 'is_train'
    }
    df = df.rename(columns=col_map)
    
    # 筛选出测试集 (is_train == 0)
    df['is_train'] = pd.to_numeric(df['is_train'], errors='coerce').fillna(1).astype(int)
    test_df = df[df['is_train'] == 0].copy()
    
    print(f"数据集加载完毕。总行数: {len(df)}, 测试集行数: {len(test_df)}")
    
    if len(test_df) == 0:
        print("数据集中没有找到测试样本 (is_train == 0)。")
        return

    # 初始化统计变量
    total_count = 0
    app_name_correct_count = 0
    category_correct_count = 0
    both_correct_count = 0
    errors = []

    # 使用tqdm显示进度条
    progress_bar = tqdm(test_df.iterrows(), total=test_df.shape[0], desc="批量测试中",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    # (这部分循环逻辑保持不变)
    for index, row in progress_bar:
        total_count += 1
        query = row['final_query']
        app_name_label = row['app_name_label']
        category_label = row['category_label']

        # 打印当前正在测试的样本
        print(f"\n--- 测试样本 {total_count}/{len(test_df)} ---")
        print(f"  Query: {query}")
        
        # 调用模型 (重试逻辑保持不变)
        tasks = [] # 初始化
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tasks = decompose_query(query)
                if isinstance(tasks, list) and len(tasks) > 0 and 'error' in tasks[0] and "Network Error" in tasks[0].get("error", ""):
                     raise requests.exceptions.RequestException(tasks[0].get("message"))
                break 
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n  查询时出错，正在重试... 错误: {e}", file=sys.stderr)
                    time.sleep(2)
                else:
                    print(f"\n  查询失败次数过多，跳过。", file=sys.stderr)
                    tasks = [{"error": "Max retries exceeded"}]
        
        # 评估并实时打印结果
        if tasks and isinstance(tasks, list) and len(tasks) > 0 and 'error' not in tasks[0]:
            predicted_task = tasks[0]
            predicted_app_name = predicted_task.get('app_name', 'None')
            predicted_category = predicted_task.get('category', 'None')
            
            app_name_match = (predicted_app_name == app_name_label)
            category_match = (predicted_category == category_label)

            if app_name_match: app_name_correct_count += 1
            if category_match: category_correct_count += 1
            
            if app_name_match and category_match:
                both_correct_count += 1
                print("  ✅ 结果完全正确")
                print(f"    - 预测与期望一致: app='{predicted_app_name}', category='{predicted_category}'")
            else:
                print("  ❌ 结果错误")
                error_details = {
                    "query": query,
                    "label": {"app_name": app_name_label, "category": category_label},
                    "prediction": {"app_name": predicted_app_name, "category": predicted_category}
                }
                errors.append(error_details)
                print(f"    - 预测: app_name='{predicted_app_name}' (匹配: {app_name_match}), category='{predicted_category}' (匹配: {category_match})")
                print(f"    - 期望: app_name='{app_name_label}', category='{category_label}'")
        else:
            print("  ❌ 模型返回错误或无效响应")
            error_details = {
                "query": query,
                "label": {"app_name": app_name_label, "category": category_label},
                "prediction": {"error": tasks[0].get('error', 'Empty or invalid response') if tasks else "Empty list"}
            }
            errors.append(error_details)
            print(f"    - 期望: app_name='{app_name_label}', category='{category_label}'")
            print(f"    - 返回: {tasks}")
            
        time.sleep(0.1) 
        
    # --- 【新增】测试循环结束后，计算并打印最终评估结果 ---
    print("\n\n" + "="*50)
    print(" " * 18 + "评估结果汇总")
    print("="*50)

    if total_count > 0:
        app_name_accuracy = (app_name_correct_count / total_count) * 100
        category_accuracy = (category_correct_count / total_count) * 100
        overall_accuracy = (both_correct_count / total_count) * 100

        print(f"测试样本总数: {total_count}")
        print(f"App Name 识别准确率: {app_name_accuracy:.2f}% ({app_name_correct_count}/{total_count})")
        print(f"Category 识别准确率: {category_accuracy:.2f}% ({category_correct_count}/{total_count})")
        print(f"整体完全匹配准确率: {overall_accuracy:.2f}% ({both_correct_count}/{total_count})")
        print(f"总错误数: {len(errors)}")
        print("="*50)

        # --- 【新增】调用函数保存结果到文件 ---
        save_results_to_file(
            model_name=MODEL_NAME,
            dataset_path=dataset_path,
            total_count=total_count,
            app_name_accuracy=app_name_accuracy,
            category_accuracy=category_accuracy,
            overall_accuracy=overall_accuracy,
            errors=errors
        )
    else:
        print("没有可供评估的测试样本。")


# --- 【新增】4. 保存评测结果的函数 ---
def save_results_to_file(model_name, dataset_path, total_count, app_name_accuracy, category_accuracy, overall_accuracy, errors, output_dir="eval_results"):
    """
    将评测结果保存到JSON文件中。

    Args:
        model_name (str): 模型名称。
        dataset_path (str): 使用的数据集路径。
        total_count (int): 测试样本总数。
        app_name_accuracy (float): app_name的准确率。
        category_accuracy (float): category的准确率。
        overall_accuracy (float): 整体准确率。
        errors (list): 错误详情列表。
        output_dir (str): 保存结果的目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 从模型名中提取一个安全的文件名部分
    safe_model_name = model_name.replace("/", "_").replace(":", "-")
    file_name = f"{safe_model_name}_eval_{timestamp}.json"
    file_path = os.path.join(output_dir, file_name)

    # 准备要保存的数据
    results_data = {
        "evaluation_metadata": {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "evaluation_time": datetime.now().isoformat()
        },
        "summary": {
            "total_samples": total_count,
            "app_name_accuracy": f"{app_name_accuracy:.2f}%",
            "category_accuracy": f"{category_accuracy:.2f}%",
            "overall_accuracy": f"{overall_accuracy:.2f}%",
            "total_errors": len(errors)
        },
        "error_details": errors
    }

    # 写入JSON文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4)
        print(f"\n评测结果已成功保存至: {file_path}")
    except Exception as e:
        print(f"\n错误：保存评测结果文件失败。 {e}", file=sys.stderr)


# --- 5. 主程序入口 (无需改动) ---
if __name__ == "__main__":
    dataset_file_path = r"/home/workspace/lgq/data/shop/generated_queries - 0704手动筛选后.csv"
    
    if not os.path.exists(dataset_file_path):
        print(f"错误：找不到指定的数据集文件 '{dataset_file_path}'", file=sys.stderr)
    else:
        batch_test_and_evaluate(dataset_file_path)