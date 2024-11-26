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

def prefill_without_prefix(model, input_ids, past_key_values):
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

def prefill_with_prefix(model, input_ids, past_key_values):
  pos = -1
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
      past_key_values=new_key_values,
      use_cache=True,
  )
  end_time = time.perf_counter()
  elapsed_time = end_time - start_time

  pred_token_ids = outputs.logits[:, -1, :].argmax(dim=-1)
  return outputs.past_key_values, elapsed_time, pred_token_ids

def run_multi_turn(func, model, input_ids, past_key_values, num):
  times = []
  pred_token_ids = None

  for _ in range(num):
    _, time, pred = func(model, input_ids, past_key_values)
    times.append(time)
    if pred_token_ids is None:
      pred_token_ids = pred.item()
    assert(pred_token_ids == pred)

  return sum(times) / len(times)

def main():
  model_path = '/data/llm/longchat-7b-v1.5-32k'
  model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto", 
                    trust_remote_code=True,
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  prompt = generate_random_prompt(200, tokenizer)
  
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

  past_key_values, _, _ = prefill_without_prefix(model, input_ids, None)

  time_0 = run_multi_turn(prefill_without_prefix, model, input_ids, past_key_values, 100)

  time_1 = run_multi_turn(prefill_with_prefix, model, input_ids, past_key_values, 100)

  print(f'time_0 = {time_0}')
  print(f'time_1 = {time_1}')

if __name__ == '__main__':
  main()