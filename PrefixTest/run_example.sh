#!/bin/bash

attn_list=("flash_attention_2" "sdpa" "eager")

for attn in "${attn_list[@]}"; do
    python example.py --attn_implementation "$attn"
done
