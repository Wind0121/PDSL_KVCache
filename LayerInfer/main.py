from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from monkey_patch import replace_llama
import warnings
import time
warnings.filterwarnings("ignore")

def load_model(model_path):
  model = LlamaForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    attn_implementation='eager',
                    device_map="auto",
                    trust_remote_code=True
                    ).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  return model, tokenizer

def greedy_generate_token(model, input_ids, use_cache=False, past_key_values=None):
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
  return logits, cur_key_values

def generate(model, tokenizer, prompt, max_len, use_cache):
  prompt = "USER: " + prompt + "\n\nASSISTANT: "
  print(f'\n{prompt}', end='')
  input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
  past_key_values = None
  generated_ids = []
  pos = 0

  for _ in range(max_len):
    logits, past_key_values = greedy_generate_token(model, input_ids, use_cache, past_key_values)
    pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids.append(pred_token_idx.item())
    input_ids = pred_token_idx if use_cache else torch.cat([input_ids, pred_token_idx], dim=1)
    generated_text = (
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        ).strip().split(" ")
    )
    now = len(generated_text) - 1
    if now > pos:
        print(" ".join(generated_text[pos:now]), end=" ", flush=True)
        pos = now

    if pred_token_idx == tokenizer.eos_token_id:
        break
  print(" ".join(generated_text[pos:]), flush=True)
  

def main():
  model_path = '/data/llm/longchat-7b-v1.5-32k'
  model_name = model_path.split('/')[-1]

  model, tokenizer = load_model(model_path)

  replace_llama(model)

  prompt = 'What is your name'

  generate(model, tokenizer, prompt, 100, True)

if __name__ == '__main__':
  main()

