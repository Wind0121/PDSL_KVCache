from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np

def evaluate_on_attentions(attentions, query_length):
  # 三组实验：
  # 1. prompt0 的 query 在 context 部分的注意力与 context 本身的注意力对比
  # 2. prompt1 的 query 在 context 部分的注意力与 context 本身的注意力对比
  # 3. prompt0 和 prompt1 的 query 在 context 部分的注意力对比
  
  assert(len(attentions) == 2)
  attentions_0, attentions_1 = attentions
  query_length_0, query_length_1 = query_length

  # 得到各层的mean值
  attentions_0 = [torch.mean(attn, dim=1)[0] for attn in attentions_0]
  attentions_1 = [torch.mean(attn, dim=1)[0] for attn in attentions_1]

  # 得到各层的 context 本身的注意力
  context_attentions_0 = [attention[-(query_length_0 + 1)][:-(query_length_0)].cpu().detach().numpy() for attention in attentions_0]
  context_attentions_1 = [attention[-(query_length_1 + 1)][:-(query_length_1)].cpu().detach().numpy() for attention in attentions_1]
  context_attentions_0 = [(attention - attention.min()) / (attention.max() - attention.min()) for attention in context_attentions_0]
  context_attentions_1 = [(attention - attention.min()) / (attention.max() - attention.min()) for attention in context_attentions_1]
  assert(context_attentions_0[0].shape[0] == context_attentions_1[0].shape[0])

  # 得到各层 query 在 context 部分的注意力
  query_attentions_0 = [attention[-1][:-(query_length_0)].cpu().detach().numpy() for attention in attentions_0]
  query_attentions_1 = [attention[-1][:-(query_length_1)].cpu().detach().numpy() for attention in attentions_1]
  query_attentions_0 = [(attention - attention.min()) / (attention.max() - attention.min()) for attention in query_attentions_0]
  query_attentions_1 = [(attention - attention.min()) / (attention.max() - attention.min()) for attention in query_attentions_1]
  assert(query_attentions_0[0].shape[0] == query_attentions_1[0].shape[0])

  # test2
  l2_dis_0 = [np.linalg.norm(context - query) for context, query in zip(context_attentions_0, query_attentions_0)]
  l2_dis_1 = [np.linalg.norm(context - query) for context, query in zip(context_attentions_1, query_attentions_1)]
  l2_dis_2 = [np.linalg.norm(query_0 - query_1) for query_0, query_1 in zip(query_attentions_0, query_attentions_1)]
  return [l2_dis_0, l2_dis_1, l2_dis_2]


def main():
  model_path = '/data/llm/Meta-Llama-3-8B-Instruct'
  data_path = 'data/NIAH.txt'
  needles = ('The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.', 'The weather in New York is cold.')
  retrieval_query = ('The best thing to do in San Francisco is: ', 'The weather in New York is: ')
  save_path = 'NIAH_result.json'

  with open(data_path, "r") as f:
    context = f.read()
  
  querys = [f"\n Based on the content of the book, Question: {query}\nAnswer:" for query in retrieval_query]

  prompts = [f"<|im_start|> This is a very long story book: <book> {context} </book>." + query for query in querys]

  model_name = model_path.split('/')[-1]
  model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    attn_implementation='flash_attention_2',
                    device_map="auto",
                    low_cpu_mem_usage=True, 
                    use_cache=True, 
                    trust_remote_code=True
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

  prompts = [tokenizer(prompt, return_tensors="pt") for prompt in prompts]
  input_ids = [prompt['input_ids'].to(model.device) for prompt in prompts]

  output_ids = [model(
                  input_id, 
                  output_attentions=False,
                  max_new_tokens=30,
                  num_beams=1,
                  do_sample=False,
                  temperature=1.0,
                  eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
                )
                for input_id in input_ids
              ]
  responses = [tokenizer.decode(output_id[0][input_id.shape[1]:], skip_special_tokens=True).strip() for output_id, input_id in zip(output_ids, input_ids)]

  querys = [tokenizer(query, return_tensor="pt") for query in querys]
  query_ids = [query['input_ids'].to(model.device) for query in querys]
  query_length = [query_id.shape[1] for query_id in query_ids]
  attentions =  [model(
                      input_id, 
                      output_attentions=True
                  ).attentions
                  for input_id in input_ids
                ]
  l2_dis = evaluate_on_attentions(attentions, query_length)

  result = {
    'model' : model_name,
    'needle' : needles,
    'model_response' : responses,
    'l2_dis_0' : l2_dis[0],
    'l2_dis_1' : l2_dis[1],
    'l2_dis_2' : l2_dis[2]
  }

  with open(save_path, 'w') as f:
    json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
  main()