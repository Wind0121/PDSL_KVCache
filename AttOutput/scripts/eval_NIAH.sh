mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/


METHOD='full'       # ['full', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o', 'cam']
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
model_path="/data/llm/Meta-Llama-3-8B-Instruct"
model_name="Meta-Llama-3-8B-Instruct"
save_file=${model_name}_${METHOD}
# For Llama3-8b

(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 2001\
    --model_provider LLaMA3 \
    --model_name ${model_path} \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --model_version ${save_file}
) 2>&1  | tee results_needle/logs/${save_file}.log