import transformers
from llama_model import get_layer_num, embedding_input, embedding_output, llama_for_causallm_forward, llama_model_forward
import types

def replace_llama(model):
  model.get_layer_num = types.MethodType(get_layer_num, model)
  model.embedding_input = types.MethodType(embedding_input, model)
  model.embedding_output = types.MethodType(embedding_output, model)
  model.forward = types.MethodType(llama_for_causallm_forward, model)
  model.model.forward = types.MethodType(llama_model_forward, model.model)
