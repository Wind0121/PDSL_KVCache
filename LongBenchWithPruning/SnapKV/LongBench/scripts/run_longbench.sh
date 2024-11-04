# example
model=longchat-v1.5-7b-32k
dataset=qasper
compress_args_path=ablation_c4096_w32_k7_maxpool.json

CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model ${model} --compress_args_path ${compress_args_path} --dataset ${dataset}
python eval.py --model ${model}