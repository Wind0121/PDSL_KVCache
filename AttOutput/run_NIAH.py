from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json


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

  result = {
    'model' : model_name,
    'needle' : needles,
    'model_response' : responses
  }

  with open(save_path, 'w') as f:
    json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
  main()