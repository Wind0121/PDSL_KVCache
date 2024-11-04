import warnings

warnings.filterwarnings("ignore")

import torch
import argparse

from utils import load, load_jsonl, draw_attention_scores

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_gen_len, past_key_values=None, use_cache=True, output_attentions=True):
    # Prefill
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions
    )

    # type(outputs) == CausalLMOutputWithPast
    past_key_values = outputs.past_key_values
    attentions = list(outputs.attentions)

    # greedy choose
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    input_ids = pred_token_idx if use_cache else torch.cat([input_ids, pred_token_idx], dim=1)
    generated_ids = [pred_token_idx.item()]
    pos = 0

    # Decode
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        past_key_values = outputs.past_key_values
        
        # if use_cache:
        #     # 使用 KVCache 的话，需要拼接新 token 的注意力矩阵
        #     attentions = [torch.nn.functional.pad(attentions[i], (0, outputs.attentions[0].shape[-1] - attentions[0].shape[-1], 0, 0, 0, 0, 0, 0), mode='constant', value=0) for i in range(len(attentions))]
        #     attentions = [torch.cat([attentions[i], outputs.attentions[i]], dim=-2) for i in range(len(attentions))]
        # else:
        #     # 不使用 KVCache 的话，输出的就是当前整个序列的注意力矩阵
        #     attentions = list(outputs.attentions)

        # greedy choose
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        input_ids = pred_token_idx if use_cache else torch.cat([input_ids, pred_token_idx], dim=1)
        generated_ids.append(pred_token_idx.item())

        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values, attentions


@torch.no_grad()
def inference(model, tokenizer, prompts, max_gen_len=1000):
    past_key_values = None
    use_cache = True
    output_attentions = False

    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        past_key_values, attentions = greedy_generate(model, tokenizer, input_ids, max_gen_len, past_key_values, use_cache, output_attentions)
        # draw_attention_scores(attentions)
        

def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    data_filepath = args.data_file
    print(f"Loading data from {data_filepath} ...")

    list_data = load_jsonl(data_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    inference(
        model,
        tokenizer,
        prompts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_file", type=str, default="data/mini_data.jsonl")
    args = parser.parse_args()

    main(args)
