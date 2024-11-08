import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import subprocess

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer

def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

def draw_attention_scores(attentions, output_path):
    # 可以指定需要输出的注意力分数矩阵
    layer_start, layer_end = 0, len(attentions)
    head_start, head_end = 0, attentions[0].shape[1]

    fig, axes = plt.subplots(layer_end - layer_start, head_end - head_start + 1, figsize=(100, 80))
    
    for layer in range(layer_start, layer_end):
        attention = attentions[layer] * 10000
        for head in range(head_start, head_end):
            ax = axes[layer - layer_start, head - head_start]  # 获取第 layer 行，第 head 列的子图
            ax.imshow(attention.cpu().detach().numpy()[0, head], vmax=100)
            ax.set_xlabel('Key Positions')
            ax.set_ylabel('Query Positions')
            ax.set_title(f'Layer {layer}, Head {head}')
        
        # draw mean attention
        attention_average = torch.mean(attention[:, head_start : head_end, :, :], dim=1)[0]
        attention_average = attention_average.cpu().detach().numpy()
        ax = axes[layer - layer_start, -1]
        ax.imshow(attention_average, vmax=100)
        ax.set_xlabel('Key Positions')
        ax.set_ylabel('Query Positions')
        ax.set_title(f'Layer {layer}, Mean')
        

    plt.tight_layout()

    plt.savefig(output_path, format='png', dpi=150)  # 文件名、格式和分辨率

def draw_single_mean_attention(all_layers_attentions, input_ids, output_directory):
    for layer_idx, attentions in enumerate(all_layers_attentions):
        attention = attentions * 10000

        attention_average = torch.mean(attention, dim=1)

        attention_average = attention_average[0]

        attention = attention_average

        import matplotlib.pyplot as plt
        import numpy as np

        attention = attention.cpu().detach().numpy()

        plt.figure(figsize=(100, 80))
    
        fig, ax = plt.subplots()
        ax.imshow(attention, vmax=100)

        plt.title(f'Attention Weights Heatmap Layer {layer_idx}')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.savefig(f'{output_directory}/layer{layer_idx}.png', dpi=150, format='png')

    save_picture_pdf(f'{output_directory}', f'{output_directory}/output.pdf', True)

def save_picture_pdf(image_folder, output_pdf_path, remove=False):
    # 定义图片文件夹和输出PDF文件名
    image_folder = image_folder  
    output_pdf = output_pdf_path

    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(('jpg', 'jpeg', 'png'))]

    # 按文件名排序（可选）
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else float('inf'))

    # 将图片转换为PDF
    image_list = []
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = Image.open(img_path)
        # 将图片转换为RGB模式
        img = img.convert('RGB')
        image_list.append(img)

    # 保存为PDF文件
    if image_list:
        image_list[0].save(output_pdf, save_all=True, append_images=image_list[1:])

    print(f'PDF文件已生成：{output_pdf}')

    if remove:
        for image in images:
            os.remove(os.path.join(image_folder, image))

def upload_file_to_oss(file_path):
    file_name = file_path.split("/")[-1]
    try:
        upload_command = [ "aliyun", "oss", "cp", file_path, f"oss://kv-cache/", "-e", "oss-accelerate.aliyuncs.com" ]
        subprocess.run(upload_command, check=True)
        url = f"https://kv-cache.oss-cn-hangzhou.aliyuncs.com/{file_name}"
        print("URL: ", url)
        return url
    except subprocess.CalledProcessError as e:
        print("OSS upload failed: ", e)
        return None