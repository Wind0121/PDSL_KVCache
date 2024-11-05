export CUDA_VISIBLE_DEVICES=0

method=PyramidKV # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM
max_capacity_prompts=2048 # 128,2048 in paper
attn_implementation=flash_attention_2 # Support "flash_attention_2", "sdpa", "eager".
source_path=/home/zk/PyramidKV/
model_path=/data/llm/llama-2-7b-hf
save_dir=${source_path}"results_long_bench" # path to result save_dir

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True
