import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np
import time

def generate_random_prompt(length: int, tokenizer):
  prompt_token_ids = np.random.randint(0, tokenizer.vocab_size, size=length).tolist()
  prompt = tokenizer.decode(prompt_token_ids)
  return prompt

def greedy_generate(model, input_ids, past_key_values=None, use_cache=False, prefix_hit_rate=1.0):
  if past_key_values is not None:
    assert(input_ids.shape[1] == past_key_values[0][0].shape[2])
    pos = min(int(input_ids.shape[1] * prefix_hit_rate), input_ids.shape[1] - 1) - input_ids.shape[1]
    input_ids = input_ids[:, pos:]
    new_key_values = []
    for key_value in past_key_values:
      key, value = key_value
      key = key[:, :, :pos, :]
      value = value[:, :, :pos, :]
      new_key_values.append((key, value))
    past_key_values = tuple(new_key_values)

  start_time = time.perf_counter()
  outputs = model(
      input_ids=input_ids,
      past_key_values=past_key_values,
      use_cache=use_cache,
  )
  end_time = time.perf_counter()
  elapsed_time = end_time - start_time

  pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1)
  return elapsed_time, pred_token_idx

def run_multi_turn(model, input_ids, past_key_values,num):
  _, pred_0 = greedy_generate(model, input_ids, None, False)

  _, pred_1 = greedy_generate(model, input_ids, past_key_values, True)

  assert(pred_0.item() == pred_1.item())

  time_0, time_1 = [], []

  for _ in range(num):
    tm, pred = greedy_generate(model, input_ids, None, False)
    assert(pred.item() == pred_0.item())
    time_0.append(tm)
  
  for _ in range(num):
    tm, pred = greedy_generate(model, input_ids, past_key_values, True)
    assert(pred.item() == pred_1.item())
    time_1.append(tm)
  
  time_0 = sum(time_0) / len(time_0) * 1000
  time_1 = sum(time_1) / len(time_1) * 1000

  return time_0, time_1

def main():
  model_path = '/data/llm/longchat-7b-v1.5-32k'
  model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto", 
                    trust_remote_code=True,
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  prompt = generate_random_prompt(2000, tokenizer)
  
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

  past_key_values = model(input_ids=input_ids, past_key_values=None, use_cache=True).past_key_values

  time_0, time_1 = run_multi_turn(model, input_ids, past_key_values, 100)

  print(f'time_0 = {time_0:.3f}ms\ntime_1 = {time_1:.3f}ms')

if __name__ == '__main__':
  main()