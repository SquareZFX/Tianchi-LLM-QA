#模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('qwen/Qwen-7B-Chat', cache_dir="Qwen-7B-Chat")

model_dir = snapshot_download('Jerry0/M3E-large', cache_dir="M3E-large")

model_dir = snapshot_download('Xorbits/bge-reranker-large', cache_dir="bge-reranker-large")