import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import utils

method = "FullKV"
model_checkpoint_path = "/data/llm/longchat-7b-v1.5-32k"
model_name = model_checkpoint_path.split('/')[-1]
model, tokenizer = utils.load(model_checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.config.output_attention = True

directory = f"assets/{model_name}/{method}"
file_name = f"{model_name}_{method}"


if not os.path.exists(directory):
    os.makedirs(directory)



def manual_infer_with_llama_with_attention(prompt, max_length=200):

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    all_layers_attentions = [] 
    generated_ids = []
    pos = 0
    print(prompt, end=" ")

    for _ in range(max_length):

        raw_outputs = model(input_ids, output_attentions=True)
        output = raw_outputs.logits
        
        attentions = raw_outputs.attentions

        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(next_token.item())
        generated_text = (
            tokenizer.decode(generated_ids, skip_special_tokens=True,)
            .strip()
            .split(" ")
        )
        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token in tokenizer.all_special_ids:
            break

    print(" ".join(generated_text[pos:]), flush=True)
        
    for i in range(len(attentions)):
        all_layers_attentions.append(attentions[i].detach().cpu())
    return tokenizer.decode(input_ids[0], skip_special_tokens=True), input_ids[0], all_layers_attentions

prompts = []
list_data = utils.load_jsonl('data/mini_data.jsonl')
for sample in list_data:
    prompts += sample["turns"]

for i, input_prompt in enumerate(prompts):
    input_prompt = "USER: " + input_prompt + "\n\nASSISTANT: "
    results, input_ids, all_layers_attentions= manual_infer_with_llama_with_attention(input_prompt)
    utils.draw_single_mean_attention(all_layers_attentions, input_ids, directory, i)

# draw_attention_scores(all_layers_attentions, f'{directory}/{file_name}_5.png')


   