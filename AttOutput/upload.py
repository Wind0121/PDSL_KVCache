from utils import upload_file_to_oss

file_path = "/home/zk/PDSL_KVCache/AttOutput/assets/Meta-Llama-3-8B-Instruct/PyramidKV/Llama-3-8B-Instruct_0.png"

print(upload_file_to_oss(file_path))

# https://kv-cache.oss-cn-hangzhou.aliyuncs.com/llama-2-7b-hf_0.png
# https://kv-cache.oss-cn-hangzhou.aliyuncs.com/longchat-7b-v1.5-32k_0.png
# https://kv-cache.oss-cn-hangzhou.aliyuncs.com/Llama-3-8B-Instruct_0.png