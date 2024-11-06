export CUDA_VISIBLE_DEVICES=0
# Define the methods to iterate over
methods=("FullKV" "PyramidKV" "SnapKV" "H2O" "StreamingLLM")

# Define the other parameters
max_capacity_prompts=2048
attn_implementation="flash_attention_2"
source_path="/home/zk/PyramidKV/"
model_path="/data/llm/Meta-Llama-3-8B-Instruct"
save_dir="${source_path}results_long_bench"
use_cache=True
max_num_examples=200

# Loop over the methods
for method in "${methods[@]}"; do
  echo "Running benchmark for method: ${method}"
  
  # Run the Python script with the current method
  python3 run_longbench.py \
      --method ${method} \
      --model_path ${model_path} \
      --max_capacity_prompts ${max_capacity_prompts} \
      --attn_implementation ${attn_implementation} \
      --save_dir ${save_dir} \
      --use_cache ${use_cache} \
      --max_num_examples ${max_num_examples}
  
  # Optionally, wait for a few seconds before running the next method
  sleep 2
done

echo "All benchmarks completed."
