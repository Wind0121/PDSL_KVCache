from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from monkey_patch import replace_llama
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")

def load_model(model_path):
  model = LlamaForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    attn_implementation='flash_attention_2',
                    device_map="auto",
                    trust_remote_code=True
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  return model, tokenizer

def layerwise_generate_token(model, input_ids, use_cache=False, past_key_values=None):
  hidden_states = model.embedding_input(input_ids)
  layer_num = model.get_layer_num()
  cur_key_values = past_key_values
  for layer_idx in range(layer_num):
    outputs = model(li_hidden_states=hidden_states,
                    layer_idx=layer_idx,
                    past_key_values=past_key_values,
                    cur_key_values=cur_key_values,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=use_cache,
                    return_dict=True
                    )
    hidden_states = outputs.hidden_states[-1]
    cur_key_values = outputs.past_key_values
  logits = model.embedding_output(hidden_states)
  outputs.logits, outputs.past_key_values, outputs.hidden_states = logits, cur_key_values, None
  return outputs

def normal_generate_token(model, input_ids, use_cache=False, past_key_values=None):
  outputs = model(input_ids=input_ids,
                  use_cache=use_cache,
                  past_key_values=past_key_values,
                  output_attentions=False,
                  output_hidden_states=False,
                  return_dict=True
                  )
  return outputs

def generate(model, generate_fn, tokenizer, prompt, max_len, use_cache, dump):
  prompt = "USER: " + prompt + "\n\nASSISTANT: "
  input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
  past_key_values = None
  generated_ids = []
  pos = 0

  for _ in range(max_len):
    outputs = generate_fn(model, input_ids, use_cache, past_key_values)
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids.append(pred_token_idx.item())
    input_ids = pred_token_idx if use_cache else torch.cat([input_ids, pred_token_idx], dim=1)
    if pred_token_idx == tokenizer.eos_token_id:
        break
  
  if dump:
    generated_text = (
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        ).strip().split(" ")
    )
    print(f'\n{prompt}', end='')
    print(" ".join(generated_text[:]), flush=True)
  

def main():
  model_path = '/data/llm/longchat-7b-v1.5-32k'
  model_name = model_path.split('/')[-1]
  epochs = 100

  model, tokenizer = load_model(model_path)

  prompt = 'What is your name'

  start_time = time.perf_counter()
  for _ in tqdm(range(epochs), desc='normal generate'):
    generate(model, generate_fn=normal_generate_token, tokenizer=tokenizer, prompt=prompt, max_len=100, use_cache=True, dump=False)
  end_time = time.perf_counter()
  elapsed_time_0 = (end_time - start_time) / epochs

  replace_llama(model)

  start_time = time.perf_counter()
  for _ in tqdm(range(epochs), desc='layerwise generate'):
    generate(model, generate_fn=layerwise_generate_token, tokenizer=tokenizer, prompt=prompt, max_len=100, use_cache=True, dump=False)
  end_time = time.perf_counter()
  elapsed_time_1 = (end_time - start_time) / epochs

  print(f'normal generate time: {elapsed_time_0:.3f}s')
  print(f'layerwise generate time: {elapsed_time_1:.3f}s')

if __name__ == '__main__':
  main()

