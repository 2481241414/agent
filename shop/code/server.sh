python -m vllm.entrypoints.openai.api_server \
    --model /home/workspace/lgq/shop/model/Qwen3-Embedding-0.6B \
    --trust-remote-code \
    --port 8000