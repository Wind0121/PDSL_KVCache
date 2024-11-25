import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import PromptTemplate

query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

Settings.llm = HuggingFaceLLM(
    model_name="/data/llm/longchat-7b-v1.5-32k",
    tokenizer_name="/data/llm/longchat-7b-v1.5-32k",
    query_wrapper_prompt=query_wrapper_prompt,
    device_map="auto",
    generate_kwargs={"do_sample": False},
    stopping_ids=[50278, 50279, 50277, 1, 0],
)

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

print(response)