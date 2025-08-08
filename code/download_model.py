import time
import os
os.environ['HTTP_PROXY'] = 'http://proxysysx.his.hihonor.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxysysx.his.hihonor.com:8080'
os.environ['http_proxy'] = 'http://proxysysx.his.hihonor.com:8080'
os.environ['https_proxy'] = 'http://proxysysx.his.hihonor.com:8080'
from huggingface_hub import snapshot_download
from huggingface_hub import login
repo_type = "model"  # model or dataset
repo_id = "Qwen/Qwen-Image"
# repo_id = "Qwen/Qwen3-4B-Instruct-2507"
# repo_id = "Qwen/Qwen3-Reranker-8B"
# repo_id = "openai/gpt-oss-120b"
local_dir = f'/home/workspace/zm/hf_model/{repo_id.split("/")[-1]}'  # model
print(f'\033[32m开始下载{repo_id}...\033[0m')
while True:
    sd_params = {
        "local_dir": local_dir,
        # "cache_dir": cache_dir,
        "repo_id": repo_id,
        "repo_type": repo_type, 
        "local_dir_use_symlinks": False,
        "resume_download": True,
        "max_workers": 16,
    }
    if repo_type == 'model':
        sd_params['all_patterns'] = ["*.model", "*.json", "*.bin", "*.safetensors", "*.py", "*.md", "*.txt", "*.cpp", "*.cu", "*.tiktoken", "*.pth",  "*.pt","*tensor*"] # "*.bin"
        sd_params['ignore_patterns'] = ["*.msgpack","*.h5", "*.ot",]
    try:
        snapshot_download(
            local_dir=local_dir,
            # cache_dir=cache_dir,
            repo_id=repo_id,
            repo_type="model", 
            # repo_type="dataset",
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=16,
            allow_patterns=["*.model", "*.json", "*.bin", "*.safetensors",
            "*.py", "*.md", "*.txt", "*.cpp", "*.cu", "*.tiktoken", "*.pth",  "*.pt","*tensor*"], # "*.bin", 
            ignore_patterns=["*.msgpack","*.h5", "*.ot",],
        )
    except Exception as e :
        print(e)
        # time.sleep(5)
    else:
        print(f'\033[31m下载完毕：{local_dir}\033[0m')
        break