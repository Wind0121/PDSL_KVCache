from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np

def get_same(A, B):
  count = 0
  for a in A:
    if a in B:
      count += 1
  return 1.0 * count / len(A)

def cosine_similarity(A, B):
    # 计算点积
    dot_product = np.dot(A, B)
    # 计算向量A和B的范数
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def find_first_diff_loc(arr0, arr1):
  for i, (a, b) in enumerate(zip(arr0, arr1)):
    if a[0] != b[0]:
      return 1.0 * i / len(arr0)
  return 1.0

def evaluate_on_attentions(attentions, query_length):
  # 三组对比：
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

  # test1
  sort_context_attentions_0 = [[(i, score) for i, score in enumerate(attention)] for attention in context_attentions_0]
  sort_context_attentions_0 = [sorted(layer_scores, key=lambda x: x[1], reverse=True) for layer_scores in sort_context_attentions_0]
  sort_context_attentions_1 = [[(i, score) for i, score in enumerate(attention)] for attention in context_attentions_1]
  sort_context_attentions_1 = [sorted(layer_scores, key=lambda x: x[1], reverse=True) for layer_scores in sort_context_attentions_1]

  sort_query_attentions_0 = [[(i, score) for i, score in enumerate(attention)] for attention in query_attentions_0]
  sort_query_attentions_0 = [sorted(layer_scores, key=lambda x: x[1], reverse=True) for layer_scores in sort_query_attentions_0]
  sort_query_attentions_1 = [[(i, score) for i, score in enumerate(attention)] for attention in query_attentions_1]
  sort_query_attentions_1 = [sorted(layer_scores, key=lambda x: x[1], reverse=True) for layer_scores in sort_query_attentions_1]

  first_diff_loc_context0_query0 = []
  first_diff_loc_context1_query1 = []
  first_diff_loc_query0_query1 = []
  for arr0, arr1 in zip(sort_context_attentions_0, sort_query_attentions_0):
    first_diff_loc_context0_query0.append(find_first_diff_loc(arr0, arr1))
  for arr0, arr1 in zip(sort_context_attentions_1, sort_query_attentions_1):
    first_diff_loc_context1_query1.append(find_first_diff_loc(arr0, arr1))
  for arr0, arr1 in zip(sort_query_attentions_0, sort_query_attentions_1):
    first_diff_loc_query0_query1.append(find_first_diff_loc(arr0, arr1))

  # test2
  l2_context0_query0 = [float(np.linalg.norm(context - query)) for context, query in zip(context_attentions_0, query_attentions_0)]
  l2_context1_query1 = [float(np.linalg.norm(context - query)) for context, query in zip(context_attentions_1, query_attentions_1)]
  l2_query0_query1 = [float(np.linalg.norm(query_0 - query_1)) for query_0, query_1 in zip(query_attentions_0, query_attentions_1)]

  # test3
  cos_context0_query0 = [float(cosine_similarity(context, query)) for context, query in zip(context_attentions_0, query_attentions_0)]
  cos_context1_query1 = [float(cosine_similarity(context, query)) for context, query in zip(context_attentions_1, query_attentions_1)]
  cos_query0_query1 = [float(cosine_similarity(query_0, query_1)) for query_0, query_1 in zip(query_attentions_0, query_attentions_1)]

  # test4
  top_context_attention_0 = [attention[:int(0.15 * len(attention))] for attention in sort_context_attentions_0]
  top_context_attention_0 = [[pt[0] for pt in attention] for attention in top_context_attention_0]
  top_context_attention_1 = [attention[:int(0.15 * len(attention))] for attention in sort_context_attentions_1]
  top_context_attention_1 = [[pt[0] for pt in attention] for attention in top_context_attention_1]
  
  top_query_attention_0 = [attention[:int(0.15 * len(attention))] for attention in sort_query_attentions_0]
  top_query_attention_0 = [[pt[0] for pt in attention] for attention in top_query_attention_0]
  top_query_attention_1 = [attention[:int(0.15 * len(attention))] for attention in sort_query_attentions_1]
  top_query_attention_1 = [[pt[0] for pt in attention] for attention in top_query_attention_1]

  inter_context0_query0 = [len(set(a) & set(b)) for a, b in zip(top_context_attention_0, top_query_attention_0)]
  inter_context1_query1 = [len(set(a) & set(b)) for a, b in zip(top_context_attention_1, top_query_attention_1)]
  inter_query0_query1 = [len(set(a) & set(b)) for a, b in zip(top_query_attention_0, top_query_attention_1)]

  return (first_diff_loc_context0_query0, first_diff_loc_context1_query1, first_diff_loc_query0_query1), (l2_context0_query0, l2_context1_query1, l2_query0_query1), \
          (cos_context0_query0, cos_context1_query1,cos_query0_query1), (inter_context0_query0, inter_context1_query1, inter_query0_query1)


def main():
  model_path = '/data/llm/longchat-7b-v1.5-32k'
  data_path = 'data/mini_NIAH.txt'
  needles = ('The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.', 'The weather in New York is cold.')
  retrieval_query = ('The best thing to do in San Francisco is: ', 'The weather in New York is: ')
  save_path = 'NIAH_result.json'

  with open(data_path, "r") as f:
    context = f.read()
  
  querys = [f"\n Based on the content of the book, Question: {query}\nAnswer:" for query in retrieval_query]

  prompts = [f"<|im_start|> This is a very long story book: <book> {context} </book>." + query for query in querys]

  model_name = model_path.split('/')[-1]
  model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    # attn_implementation='flash_attention_2',
                    device_map="auto",
                    low_cpu_mem_usage=True, 
                    use_cache=True, 
                    trust_remote_code=True
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  prompts = [tokenizer(prompt, return_tensors="pt") for prompt in prompts]
  input_ids = [prompt['input_ids'].to(model.device) for prompt in prompts]

  output_ids = [model.generate(
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

  querys = [tokenizer(query, return_tensors="pt") for query in querys]
  query_ids = [query['input_ids'].to(model.device) for query in querys]
  query_length = [query_id.shape[1] for query_id in query_ids]
  attentions = [model(input_id, output_attentions=True).attentions for input_id in input_ids]
  first_diff, l2, cos, inter = evaluate_on_attentions(attentions, query_length)

  result = {
    'model' : model_name,
    'needle' : needles,
    'model_response' : responses,
    'first_diff_loc_context0_query0' : first_diff[0],
    'first_diff_loc_context1_query1' : first_diff[1],
    'first_diff_loc_query0_query1' : first_diff[2],
    'l2_context0_query0' : l2[0],
    'l2_context1_query1' : l2[1],
    'l2_query0_query1' : l2[2],
    'cos_context0_query0' : cos[0],
    'cos_context1_query1' : cos[1],
    'cos_query0_query1' : cos[2],
    'inter_context0_query0' : inter[0],
    'inter_context1_query1' : inter[1],
    'inter_query0_query1' : inter[2]
  }

  with open(save_path, 'w') as f:
    json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
  main()