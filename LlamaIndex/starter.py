import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import PromptTemplate

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="/data/llm/bge-small-en-v1.5")

# select hf model
# models already run: [longchat-7b-v1.5-32k, Qwen2.5-7B-Instruct, stablelm-tuned-alpha-3b$]
# you can select other parameter
Settings.llm = HuggingFaceLLM(
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="/data/llm/longchat-7b-v1.5-32k",
    model_name="/data/llm/longchat-7b-v1.5-32k",
    device_map="auto",
)

# load data
documents = SimpleDirectoryReader("./data/example").load_data()
# build index
index = VectorStoreIndex.from_documents(documents)
# build query_engine which use llm above
query_engine = index.as_query_engine()
# chat
response = query_engine.query("What did the author do growing up?")
print(response)