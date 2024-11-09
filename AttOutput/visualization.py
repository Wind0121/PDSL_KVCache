import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from utils import load, save_picture_pdf, draw_attention_scores

method = "StreamingLLM"
model_checkpoint_path = "/data/llm/longchat-7b-v1.5-32k"
model_name = model_checkpoint_path.split('/')[-1]
model, tokenizer = load(model_checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.config.output_attention = True

directory = f"assets/{model_name}/{method}"
file_name = f"{model_name}_{method}"


if not os.path.exists(directory):
    os.makedirs(directory)



def manual_infer_with_llama_with_attention(prompt, max_length=1000):

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

input_prompt = "USER: Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.\n\nAnswer: "
# input_prompt = """
# [INST] <<SYS>>You are given some documents, and you need to answer a question based on these documents.Your answer should be less than five words.    
# <</SYS>>
# Document: Roman Republic After having declined in size following the subjugation of the Mediterranean, the Roman navy underwent short-term upgrading and revitalisation in the late Republic to meet several 
# new demands. Under Caesar, an invasion fleet was assembled in the English Channel to allow the invasion of Britannia; under Pompey, a large fleet was raised in the Mediterranean Sea to clear the sea of Cili
# cian pirates. During the civil war that followed, as many as a thousand ships were either constructed or pressed into service from Greek cities. 
# Document: North Sea The North Sea is bounded by the Orkney Islands and east coast of Great Britain to the west and the northern and central European mainland to the east and south, including Norway, Denmark
# , Germany, the Netherlands, Belgium, and France. In the southwest, beyond the Straits of Dover, the North Sea becomes the English Channel connecting to the Atlantic Ocean. In the east, it connects to the Ba
# ltic Sea via the Skagerrak and Kattegat, narrow straits that separate Denmark from Norway and Sweden respectively. In the north it is bordered by the Shetland Islands, and connects with the Norwegian Sea, w
# hich lies in the very north - eastern part of the Atlantic. 
# Document: Rhine The Rhine (Romansh: Rein, German: Rhein, French: le Rhin, Dutch: Rijn) is a European river that begins in the Swiss canton of Graubünden in the southeastern Swiss Alps, forms part of the Swi
# ss-Austrian, Swiss-Liechtenstein border, Swiss-German and then the Franco-German border, then flows through the Rhineland and eventually empties into the North Sea in the Netherlands. The biggest city on th
# e river Rhine is Cologne, Germany with a population of more than 1,050,000 people. It is the second-longest river in Central and Western Europe (after the Danube), at about 1,230 km (760 mi),[note 2][note 1
# ] with an average discharge of about 2,900 m3/s (100,000 cu ft/s). 
# Question: Who sent naval ships to the body of water that joins the Atlantic and the sea where the Rhine ends? 
# Answer:  [/INST]
# """

results, input_ids, all_layers_attentions= manual_infer_with_llama_with_attention(input_prompt)

draw_attention_scores(all_layers_attentions, f'{directory}/{file_name}_5.png')


   