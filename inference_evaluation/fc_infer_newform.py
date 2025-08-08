import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# --- OpenAI SDK配置 ---
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
MODEL_NAME = "/home/workspace/EasyR1/checkpoints/fc0801_r1/qwen3_4b_instruct_dapo/global_step_140/actor/huggingface"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 日志配置
model_name = "fc0801_dapo"
date = "20250805"
LOG_DIR = '../logs/infer'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_file = os.path.join(LOG_DIR, f"{date}_fc_{model_name}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 文件路径配置
origin_dir = "/home/workspace/nl/FC/data/input/fc_evaluation.json"
predict_save_path = f'/home/workspace/nl/FC/data/output/toolcall/lgq_4b/{date}'
if not os.path.exists(predict_save_path):
    os.makedirs(predict_save_path)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
save_path = os.path.join(predict_save_path,f"fc_evaluation_with_response_newform_{model_name}_{timestamp}.json")

class LLMInferOpenAI:
    def __init__(self):
        logger.info("LLMInferOpenAI initialized.")

    def preprocess(self, df_samples):
        logger.info("Preprocessing dialogue content...")
        for i in tqdm(range(len(df_samples)), desc="预处理对话"):
            convs = df_samples.at[i, 'dialogue']
            for j, conv in enumerate(convs):
                if conv['role'] in ['user', 'tool']:
                    conv['content'] = conv['content'] + ' /no_think'
                elif conv['role'] == 'system':
                    assert j == 0
                elif conv['role'] == 'assistant':
                    continue
                else:
                    logger.error(f"Unknown role: {conv['role']}")
                    raise Exception('error')
                convs[j] = conv
            df_samples.at[i, 'dialogue'] = convs
        logger.info("Preprocessing completed.")
        return df_samples

    def prepare_messages_and_labels(self, df_samples):
        logger.info("Preparing messages and labels...")
        tags, messages, labels = [], [], []
        for i in tqdm(range(len(df_samples)), desc="构建消息和标签"):
            message = []
            for msg in df_samples.at[i, 'dialogue']:
                message.append(msg)
                if msg['role'] == 'user':
                    messages.append(message.copy())
                elif msg['role'] == 'assistant':
                    labels.append(msg['content'])
                    tags.append(df_samples.at[i, 'type'])
        data = {
            'tag': tags,
            'messages': messages,
            'label': labels
        }
        df_new = pd.DataFrame(data)
        logger.info(f"Prepared dataframe with {len(df_new)} samples for inference.")
        return df_new

    def call_openai_api(self, messages):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.05,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return f"ERROR: {e}"

    def infer(self, df_samples, max_workers=16000000):
        df_samples = self.preprocess(df_samples)
        df_new = self.prepare_messages_and_labels(df_samples)

        predicts = [None] * len(df_new)
        logger.info(f"Starting OpenAI concurrent inference, max_workers={max_workers} ...")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.call_openai_api, messages): idx
                for idx, messages in enumerate(df_new['messages'])
            }
            with tqdm(total=len(df_new), desc="OpenAI并发推理", ncols=100) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        predicts[idx] = future.result()
                    except Exception as e:
                        predicts[idx] = f"ERROR: {e}"
                        logger.error(f"推理异常: idx={idx}, error={e}")
                    pbar.update(1)
        elapsed = time.time() - start_time
        qps = len(df_new) / elapsed if elapsed > 0 else float('inf')
        logger.info(f"推理完成，总耗时: {elapsed:.2f} 秒，QPS: {qps:.2f}")
        df_new['predict'] = predicts
        return df_new, elapsed, qps

if __name__ == "__main__":
    # 推理
    logger.info(f"Loading input data from {origin_dir}")
    origin_data = pd.read_json(origin_dir)
    logger.info(f"Input data loaded. Number of samples: {len(origin_data)}")

    model = LLMInferOpenAI()
    predict_file, elapsed, qps = model.infer(origin_data, max_workers=16)

    # 保存推理结果
    predict_file[['tag', 'messages', 'label', 'predict']].to_json(
        save_path,
        indent=2,
        orient='records',
        force_ascii=False
    )
    logger.info(f"推理结果已保存至: {save_path}")
    logger.info(f"推理总耗时：{elapsed:.2f} 秒，QPS：{qps:.2f}")