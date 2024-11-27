import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np
import time
import matplotlib.pyplot as plt

def save_data_to_file(time_list):
  with open('data.txt', mode="w+") as f:
    for times in time_list:
      for item in times:
        f.write(f'{item:.3f}, ')
      f.write(f'\n')

def draw_data(time_list):
  x = [100, 500, 1000, 1500, 2000]
  y0, y1, y2, y3, y4, y5 = time_list[0], time_list[1], time_list[2], time_list[3], time_list[4], time_list[5]

  plt.plot(x, y0, label="prefix-hit-rate: 0%", marker='o')
  plt.plot(x, y1, label="prefix-hit-rate: 20%", marker='s')
  plt.plot(x, y2, label="prefix-hit-rate: 40%", marker='^')
  plt.plot(x, y3, label="prefix-hit-rate: 60%", marker='v')
  plt.plot(x, y4, label="prefix-hit-rate: 80%", marker='*')
  plt.plot(x, y5, label="prefix-hit-rate: 100%", marker='D')
  
  plt.xlabel('Prompt Len')
  plt.ylabel('TTFT(ms)')

  plt.legend()

  plt.savefig('TTFT_Hit.png', dpi=300, bbox_inches='tight')

def generate_random_prompt(length: int, tokenizer):
  prompt_token_ids = np.random.randint(0, tokenizer.vocab_size, size=length).tolist()
  prompt = tokenizer.decode(prompt_token_ids)
  return prompt

def greedy_generate(model, input_ids, past_key_values=None, use_cache=False, prefix_hit_rate=1.0):
  if not use_cache:
    past_key_values = None

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

def run_multi_turn(model, input_ids, use_cache, prefix_hit_rate, num):
  #-------------------------冷启动------------------------------
  outputs = model(input_ids=input_ids, past_key_values=None, use_cache=True)
  past_key_values = outputs.past_key_values
  correct_pred = outputs.logits[:, -1, :].argmax(dim=-1)
  #-------------------------------------------------------------

  times = []
  for _ in range(num):
    tm, pred = greedy_generate(model, input_ids, past_key_values, use_cache, prefix_hit_rate)
    assert(pred.item() == correct_pred.item())
    times.append(tm)
  
  return sum(times) / len(times) * 1000

def main():
  turn_num = 1
  model_path = '/data/llm/longchat-7b-v1.5-32k'
  model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto", 
                    trust_remote_code=True,
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  input_ids_list = []
  for length in [100, 500, 1000, 1500, 2000]:
    prompt = generate_random_prompt(length, tokenizer)  
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_ids_list.append(input_ids)

  time_list = [[], [], [], [], [], []]
  for input_ids in input_ids_list:
    tm = run_multi_turn(model, input_ids, False, 0.0, turn_num)
    time_list[0].append(tm)

  for i, hit_rate in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
    for input_ids in input_ids_list:
      tm = run_multi_turn(model, input_ids, True, hit_rate, turn_num)
      time_list[i + 1].append(tm)

  save_data_to_file(time_list)

  draw_data(time_list)

if __name__ == '__main__':
  main()