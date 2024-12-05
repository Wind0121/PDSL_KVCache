import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
  add_start_docstrings,
  add_start_docstrings_to_model_forward,
  is_flash_attn_greater_or_equal_2_10,
  logging,
  replace_return_docstrings,
)

from transformers.models.llama.modeling_llama import (
  LLAMA_INPUTS_DOCSTRING,
  _CONFIG_FOR_DOC
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

logger = logging.get_logger(__name__)

def get_layer_num(self):
  return len(self.model.layers)

def embedding_input(self, input_ids):
  embed_tokens = self.get_input_embeddings()
  return embed_tokens(input_ids)

def embedding_output(self, hidden_states):
  lm_head = self.get_output_embeddings()
  return lm_head(hidden_states)

def llama_for_causallm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    li_hidden_states: Optional[torch.FloatTensor] = None,
    layer_idx: Optional[int] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    
    assert(li_hidden_states is not None)
    assert(layer_idx >= 0)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        li_hidden_states=li_hidden_states,
        layer_idx=layer_idx,
    )

    return CausalLMOutputWithPast(
        past_key_values=outputs.past_key_values,
        hidden_states=(outputs.last_hidden_state,), # 符合输出格式
    )

def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    li_hidden_states: Optional[torch.FloatTensor] = None,
    layer_idx: Optional[int] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    
    assert(li_hidden_states is not None)
    assert(layer_idx >= 0)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # inputs_embeds: (batch_size, seq_len, hidden_size)
    # 这里要么是上一层的输出，要么是自己 embedding 的
    inputs_embeds = li_hidden_states

    # 4.43 后 past_key_values 开始采用 Cache 类而不是 tuple，这里会进行转换，因为注意力层需要使用 Cache 格式
    return_legacy_cache = False
    if (
        use_cache and not isinstance(past_key_values, Cache) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    # cache_position: (seq_len, )
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    # position_ids: (1, seq_len)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # (batch_size, 1, query_length, key_value_length)
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    next_decoder_cache = None

    decoder_layer = self.layers[layer_idx]

    layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=causal_mask, # 包含了因果掩码和传入的注意力掩码 
        position_ids=position_ids,
        past_key_value=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    logger.info(f'Decoding layer idx {layer_idx}')

    # 得到这一层的输出，同时也是下一层的输入
    hidden_states = layer_outputs[0]

    if use_cache:
        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

    # 最后输出还要经过一次 norm
    if layer_idx == len(self.layers) - 1:
        hidden_states = self.norm(hidden_states)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        # 如果传入是 tuple 形式的，返回也是 tuple 形式的
        next_cache = next_cache.to_legacy_cache()

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
    )