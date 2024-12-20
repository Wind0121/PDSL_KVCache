"""
This script is adapted from 
https://github.com/FranxYao/Long-Context-Data-Engineering
"""

import tiktoken
import os 
import pdb
import glob
import jieba

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from anthropic import Anthropic
import numpy as np
import argparse
from rouge_score import rouge_scorer

import sys
import os
import tensor_parallel as tp

from openai import OpenAI
from datetime import datetime, timezone
import time
import torch

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle_num=2,
                 needle=("\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n", "\nThe weather in New York is cold.\n"),
                 haystack_dir="/home/zk/PyramidKV/data/PaulGrahamEssays", # PaulGrahamEssays  
                 retrieval_question=("The best thing to do in San Francisco is: ", "The weather in New York is: "), 
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 model_version=None, 
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True, 
                 step=100, 
                 method='full', 
                 attn_implementation='flash_attention_2',
                 output_attentions=True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        if not isinstance(needle, tuple) or not isinstance(retrieval_question, tuple):
            raise ValueError("needle and retrieval_question should be tuple")

        # 暂时只支持两个 needle
        if needle_num > 1:
            assert(needle_num == 2) #

        self.needle_num = needle_num
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.method = method
        self.attn_implementation = attn_implementation
        self.output_attentions = output_attentions

        self.model_version = model_version
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths


        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if(self.model_provider in ["LLaMA3", "Mistral"]):
            self.enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # self.enc.add_special_tokens({'pad_token': '[PAD]'})
            print("loading from %s" % model_name)


            # if torch.cuda.device_count()>1:

            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            from transformers import AutoConfig 


            if self.method == 'full':
                self.model_to_test=AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    attn_implementation=self.attn_implementation,
                    device_map="auto",
                    low_cpu_mem_usage=True, 
                    use_cache=True
                    ).eval()

        else:raise ValueError("model_provider must be either 'LLaMA3' or 'Mistral'")
            

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format = [f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {question}\nAnswer:" for question in self.retrieval_question]
            querys = [f"\n Based on the content of the book, Question: {question}\nAnswer:" for question in self.retrieval_question]
            return (tuple(test_format), tuple(querys))
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]
        
    def evaluate_on_attentions(self, attentions, query_length):
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


    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompts, querys = self.generate_prompt(context)

        assert(isinstance(prompts, tuple))

        test_start_time = time.time()
        
        if(self.model_provider in ["LLaMA3", "Mistral"]):

            prompts = [self.enc(prompt, return_tensors="pt") for prompt in prompts]
            input_ids = [prompt['input_ids'].to(self.model_to_test.device) for prompt in prompts]
            
            output_ids = [self.model_to_test.generate(
                input_id, 
                output_attentions=False,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]]
            )
            for input_id in input_ids
            ]
            responses = [self.enc.decode(output_id[0][input_id.shape[1]:], skip_special_tokens=True).strip() for output_id, input_id in zip(output_ids, input_ids)]

            if self.output_attentions and self.needle_num > 1:
                querys = [self.enc(query, return_tensor="pt") for query in querys]
                query_ids = [query['input_ids'].to(self.model_to_test.device) for query in querys]
                query_length = [query_id.shape[1] for query_id in query_ids]

                attentions = [self.model_to_test(
                    input_id, 
                    output_attentions=True
                ).attentions
                for input_id in input_ids
                ]

                l2_dis = self.evaluate_on_attentions(attentions, query_length)

        
        print(responses)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        # if len(responses) != 0:
        #     score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
        # else:
        #     score = 0.0

        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : responses,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
            'l2_dis_0' : l2_dis[0],
            'l2_dis_1' : l2_dis[1],
            'l2_dis_2' : l2_dis[2]
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Response: {responses}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            if not os.path.exists('results_needle/contexts'):
                os.makedirs('results_needle/contexts')

            if not os.path.exists(f'results_needle/contexts/{self.model_version}'):
                os.makedirs(f'results_needle/contexts/{self.model_version}')

            with open(f'results_needle/contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results_needle/results'):
                os.makedirs('results_needle/results')
            
            if not os.path.exists(f'results_needle/results/{self.model_version}'):
                os.makedirs(f'results_needle/results/{self.model_version}')

            # Save the result to file for retesting
            p = f'results_needle/results/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f, ensure_ascii=False)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results_needle/results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = tuple([self.encode_text_to_tokens(ndl) for ndl in self.needle])
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        len_tokens_needle = 0
        for ndl in tokens_needle:
            len_tokens_needle += len(ndl)

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len_tokens_needle > context_length:
            tokens_context = tokens_context[:context_length - len_tokens_needle]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle[0]
            if self.needle_num > 1:
                tokens_new_context = tokens_needle[1] + tokens_new_context
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point_0 = int(len(tokens_context) * (depth_percent / 100))
            insertion_point_1 = int(len(tokens_context) * (1 - depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context_0 = tokens_context[:insertion_point_0]
            tokens_new_context_1 = tokens_context[:insertion_point_1]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "LLaMA3"): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context_0 and tokens_new_context_0[-1] not in period_tokens:
                insertion_point_0 -= 1
                tokens_new_context_0 = tokens_context[:insertion_point_0]

            while tokens_new_context_1 and tokens_new_context_1[-1] not in period_tokens:
                insertion_point_1 -= 1
                tokens_new_context_1 = tokens_context[:insertion_point_1]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            if self.needle_num == 1:
                tokens_new_context = tokens_context[:insertion_point_0] + tokens_needle[0] + tokens_context[insertion_point_0:]
            else:
                if insertion_point_0 < insertion_point_1:
                    tokens_new_context = tokens_context[:insertion_point_0] + tokens_needle[0] + tokens_context[insertion_point_0:insertion_point_1] + tokens_needle[1] + tokens_context[insertion_point_1:]
                else:
                    tokens_new_context = tokens_context[:insertion_point_1] + tokens_needle[1] + tokens_context[insertion_point_1:insertion_point_0] + tokens_needle[0] + tokens_context[insertion_point_0:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return len(self.enc.encode(context))
        else:
            return len(self.enc.encode(context))
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(context)
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.decode(tokens[:context_length])
        else:
            return self.enc.decode(tokens[:context_length])
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "None"])
    parser.add_argument('--model_version', type=str, default=None, help='provider of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--method', type=str, default="full")
    args = parser.parse_args()

    

    ht = LLMNeedleHaystackTester(model_name=args.model_name, 
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 model_version=args.model_version, 
                                 context_lengths_min=args.s_len,
                                 save_contexts=True,
                                 save_results=True,
                                 openai_api_key=args.api_key, 
                                 context_lengths_max=args.e_len, 
                                 step=args.step, 
                                 method=args.method, 
                                 attn_implementation=args.attn_implementation,
                                 output_attentions=True
                                 )

    ht.start_test(args)