import json
import sys

# ----- 只需在这里修改文件路径 -----
input_path  = r'D:\Agent\data\20250714_sft_train_dataset.json'
output_path = r'D:\Agent\data\20250714_sft_train_dataset.jsonl'
# -----------------------------------

def json_to_jsonl(in_path: str, out_path: str):
    try:
        with open(in_path, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
    except Exception as e:
        print(f"读取 JSON 文件失败：{e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(out_path, 'w', encoding='utf-8') as f_out:
            if isinstance(data, list):
                for item in data:
                    # **每个元素独立成行**
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                # 整体写入一行
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"写入 JSONL 文件失败：{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    print(f"正在将 **{input_path}** 转换为 **{output_path}** …")
    json_to_jsonl(input_path, output_path)
    print("转换完成！")