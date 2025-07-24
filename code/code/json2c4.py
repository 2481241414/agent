import json

# 1. 从文件读取 JSON 列表
with open(r'D:\Agent\data\20250714_sft_train_dataset.json', 'r', encoding='utf-8') as f:
    src = json.load(f)          # 得到一个 list[dict]

# 2. 转换并写出
with open(r'D:\Agent\data\20250714_sft_train_dataset_c4.jsonl', 'w', encoding='utf-8') as out_f:
    for item in src:
        user_turn = f"[用户输入]\n{item['input']}"
        assistant_turn = item['output']
        text = f"{item['instruction']}\n{user_turn}\n{assistant_turn}"
        out_f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

print("转换完成，已生成 converted.jsonl")