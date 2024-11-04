import os
# CUDAVISIBLE DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers

from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral

replace_llama()

from fastchat.model import load_model, get_conversation_template

model = AutoModelForCausalLM.from_pretrained(
    "lmsys/longchat-7b-v1.5-32k",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    use_cache=True,
    use_flash_attention_2=True
    )

tokenizer = AutoTokenizer.from_pretrained("lmsys/longchat-7b-v1.5-32k", trust_remote_code=True,)

with open('snapkv.txt', 'r') as f:
    content = f.read().strip()

question = "\n What is the repository of SnapKV?"

conv = get_conversation_template("lmsys/longchat-7b-v1.5-32k")
conv.messages = []
conv.append_message(conv.roles[0],content + question)
# conv.append_message(conv.roles[0],"Who is Kobe Bryant?")
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

input_ids = tokenizer.encode(prompt, return_tensors='pt')

input_ids_len = input_ids.size(1)
print(input_ids_len)
print(input_ids.shape)

outputs = model.generate(input_ids.to(model.device), max_new_tokens=200, do_sample=False)

print(tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True))