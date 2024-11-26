import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np
import time

def generate_random_prompt(len: int, tokenizer):
  prompt_token_ids = np.random.randint(0, tokenizer.vocab_size, size=len).tolist()
  prompt = tokenizer.decode(prompt_token_ids)
  return prompt

def prefill_without_prefix(model, input_ids):
  start_time = time.perf_counter()
  outputs = model(
      input_ids=input_ids,
      past_key_values=None,
      use_cache=True,
  )
  end_time = time.perf_counter()
  elapsed_time = end_time - start_time

  pred_token_ids = outputs.logits[:, -1, :].argmax(dim=-1)
  return outputs.past_key_values, elapsed_time, pred_token_ids

def prefill_with_prefix(model, input_ids, past_key_values, prefix_hit_rate=1.0):
  pos = int(min(int(input_ids.shape[1] * prefix_hit_rate), input_ids.shape[1] - 1) - input_ids.shape[1])
  pos = -input_ids.shape[1]

  # 选取最后一个 token 作为 query
  input_ids = input_ids[:][pos:]
  # 选取除最后一个 token 的 KV Cache
  new_key_values = []
  for key_value in past_key_values:
    key, value = key_value
    key = key[:, :, :pos, :]
    value = value[:, :, :pos, :]
    new_key_values.append((key, value))
  new_key_values = tuple(new_key_values)

  start_time = time.perf_counter()
  outputs = model(
      input_ids=input_ids,
      past_key_values=None,
      use_cache=True,
  )
  end_time = time.perf_counter()
  elapsed_time = end_time - start_time

  pred_token_ids = outputs.logits[:, -1, :].argmax(dim=-1)
  return outputs.past_key_values, elapsed_time, pred_token_ids

def main():
  model_path = '/data/llm/longchat-7b-v1.5-32k'
  model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto", 
                    trust_remote_code=True,
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  prompt = generate_random_prompt(500, tokenizer)
  
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

  past_key_values, time_0, pred_0 = prefill_without_prefix(model, input_ids)

  _, time_1, pred_1 = prefill_without_prefix(model, input_ids)

  _, time_2, pred_2 = prefill_without_prefix(model, input_ids)

  # _, time_1, pred_1 = prefill_with_prefix(model, input_ids, past_key_values, 1.0)

  assert(pred_0.item() == pred_1.item())

  print(f'time_0 = {time_0}')
  print(f'time_1 = {time_1}')
  print(f'time_2 = {time_2}')

if __name__ == '__main__':
  main()