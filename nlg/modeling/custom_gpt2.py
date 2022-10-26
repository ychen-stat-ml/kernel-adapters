# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import os
import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from .custom_modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gpt2 import GPT2Config

from adapters import AdapterController
from hypercomplex.layers  import  PHMLinear
from hypercomplex.layers  import  glorot_uniform, glorot_normal

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, adapter_config=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.train_task_adapters = config.train_task_adapters and adapter_config.add_adapter_in_self_attention and not is_cross_attention
        if self.train_task_adapters:
            self.parallel_adapter = adapter_config.parallel_adapter
        
        self.train_kernel_adapter = config.kernel_adapter
        if self.train_kernel_adapter:
            self.kerfit_variant_type = adapter_config.kerfit_variant_type
        
        if adapter_config:
            self.lora = adapter_config.lora
            self.lora_head = adapter_config.lora_head
            self.lora_head_base = adapter_config.lora_head_base
            
            self.prefix_tuning = adapter_config.prefix_tuning
            self.prefix_dim = adapter_config.prefix_dim
            self.preseqlen = adapter_config.prefix_length
            self.init_prefix_from_vocab = adapter_config.init_prefix_from_vocab
            
        else:
            self.lora = ""
            self.prefix_tuning = ""
            
        if self.train_task_adapters:
            adapter_config.reduction_factor = adapter_config.task_reduction_factor
            if self.train_kernel_adapter:
                adapter_config = copy.deepcopy(adapter_config)
                adapter_config.kernel_adapter = False
            self.adapter_controller = AdapterController(adapter_config)
        
        # Lora is exclusive to adapter, while similar to adapter that it can be compatible with kernel adapter
        if self.lora != "":
        
            assert all([i in ["Q", "K", "V", "O", "C"] for i in self.lora])
        
            adapter_config.reduction_factor = adapter_config.task_reduction_factor
            if self.train_kernel_adapter:
                adapter_config = copy.deepcopy(adapter_config)
                adapter_config.kernel_adapter = False
            self.adapter_controller_l = AdapterController(adapter_config)
        
        if self.train_kernel_adapter:
            # assert not self.train_task_adapters
            adapter_config.n_head = config.n_head
            adapter_config.kernel_adapter = True
            self.adapter_controller_k = AdapterController(adapter_config)

        ###############################################################
        if adapter_config:
            self.prefix_value_extension = adapter_config.prefix_value_extension
            if adapter_config.prefix_head_share:
                self.prefix_value_extension = True
            
            self.prefix_gating_ablation = adapter_config.prefix_gating_ablation
            # self.prefix_perturbation = adapter_config.prefix_perturbation
            adapter_config.n_head = config.n_head
            
        if self.prefix_tuning == "landmark":
            if self.lora:
                adapter_config = copy.deepcopy(adapter_config)
                adapter_config.lora = ""
            
            self.adapter_controller_prefix = AdapterController(adapter_config)
            
        elif self.prefix_tuning == "MH_adapter" or self.prefix_tuning == "value_gating":
            self.adapter_controller_prefix = AdapterController(adapter_config)
            
        ###############################################################
        
        self.match_n_layer = config.n_layer
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):


        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))


        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # attn_output = torch.matmul(attn_weights, value)
        attn_weights = torch.matmul(attn_weights, value)

        # return attn_output, attn_weights
        # return attn_output, None
        return attn_weights, None
    
    def _attn_prefix(self, query, key, value,
                            attention_mask=None, head_mask=None, task=None):
        # save the computation for prefix_weights
                
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.preseqlen >= 2: 
            prefix_weights = query + self.adapter_controller_prefix((query, None), task)
            prefix_weights = (query * prefix_weights).sum(dim=3, keepdim=True)
        else:
            # (batch, head, seq_length, head_features)
            prefix_weights = (query * query).sum(dim=3, keepdim=True)
        
        # adaptive length
        if self.preseqlen == 0:
            n = prefix_weights.shape[2]
            prefix_weights = prefix_weights * torch.sqrt(torch.arange(1, n+1, device=prefix_weights.device)).reshape(1,1,-1,1)
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)
            prefix_weights = prefix_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        
        # Additional O(n^2) to compute prefix_weights
        prefix_weights = torch.cat([prefix_weights, attn_weights], dim=3) # bhn'd
        prefix_weights = nn.Softmax(dim=-1)(prefix_weights)
        if self.prefix_gating_ablation:
            attn_weights = prefix_weights[:,:,:,1:]
        else:
            attn_weights = nn.Softmax(dim=-1)(attn_weights)
        prefix_weights = prefix_weights[:,:,:,0]
        
        ##################################################
        
        attn_weights = self.attn_dropout(attn_weights)
        # prefix_output = self.attn_dropout(prefix_output)
        prefix_weights = self.attn_dropout(prefix_weights)
        
        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            # prefix_output = prefix_output * head_mask

        # attn_output = torch.matmul(attn_weights[:,:,:, 1:], value)
        attn_output = torch.matmul(attn_weights, value)
        
        if self.prefix_tuning and self.prefix_value_extension:
            prefix_output = self.adapter_controller_prefix((query, prefix_weights), task)
            return attn_output, prefix_output
        else:
            prefix_output = self.adapter_controller_prefix(query, task)
            prefix_output = torch.einsum("bhn,bhnd->bhnd", prefix_weights, prefix_output)
            return attn_output + prefix_output, None

    def _attn_prefix_gating(self, query, key, value,
                            attention_mask=None, head_mask=None, task=None):
        # save the computation for prefix_weights
        
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        # (batch, head, seq_length, head_features)
        prefix_weights = (query * query).sum(dim=3, keepdim=True)
        # adaptive length
        if self.preseqlen == 0:
            n = prefix_weights.shape[2]
            prefix_weights = prefix_weights * torch.sqrt(torch.arange(1, n+1, device=prefix_weights.device)).reshape(1,1,-1,1)
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)
            prefix_weights = prefix_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        
        ##################################################
        
        prefix_weights = torch.cat([prefix_weights, attn_weights], dim=3) # bhnd'
        prefix_weights = nn.Softmax(dim=-1)(prefix_weights)[:,:,:,0]
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        
        attn_weights = self.attn_dropout(attn_weights)
        
        ##################################################        
        
        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value)
        attn_weights = torch.einsum("bhm,bhmn->bhmn", 1-prefix_weights, attn_weights)
        
        if self.prefix_tuning and self.prefix_value_extension:
            prefix_output = self.adapter_controller_prefix((query, key, prefix_weights, attn_weights), task)
            return attn_output, prefix_output[0] + prefix_output[1]
        else:
            prefix_output = self.adapter_controller_prefix((query, key), task)  
            prefix_output = torch.einsum("bhn,bhnd->bhnd", prefix_weights, prefix_output[0])
            attn_output = attn_output + torch.matmul(attn_weights, prefix_output[1])
            
            return attn_output + prefix_output, None
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        task=None,
    ):
        if self.train_kernel_adapter and self.kerfit_variant_type == "kerfit_X":
            hidden_states, _ = self.adapter_controller_k((hidden_states, None), task)
            
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        
        if self.lora and not self.lora_head: # original lora
            query, value = self.adapter_controller_l((hidden_states, key, query, value), task)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
            
        # Scale query and key using kernel adapter
        if self.train_kernel_adapter and self.kerfit_variant_type != "kerfit_X":
            query, key = self.adapter_controller_k((query, key), task)    

        if self.lora and self.lora_head:
            # if self.lora_head_base == "X":
                # query, value = self.adapter_controller_l((hidden_states, query, value), task)
            # elif self.lora_head_base in ["K", "QS"]:
                # query, value = self.adapter_controller_l((key, query, value), task)
            query, value = self.adapter_controller_l((hidden_states, key, query, value), task)

        if self.prefix_tuning == "landmark": 
            # prefix_v = self.adapter_controller_prefix(query, task)
            attn_output, attn_weights = self._attn_prefix(query, key, value, 
                                # prefix_v, attention_mask, head_mask)
                                attention_mask, head_mask, task)
            if self.prefix_value_extension: prefix_output = attn_weights
        elif self.prefix_tuning == "value_gating":
            attn_output, prefix_output = self._attn_prefix_gating(query, key, value, 
                                attention_mask, head_mask, task)
            
        elif self.prefix_tuning == "MH_adapter":
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            
            if self.prefix_value_extension:
                prefix_output = self.adapter_controller_prefix((query, None), task)
            else:
                attn_output = attn_output + self.adapter_controller_prefix(query, task)
            
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        # if "O" in self.lora and self.lora_head:
        if "O" in self.lora:
            tmp = attn_output
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            attn_output = self.c_proj(attn_output)
            _, attn_output = self.adapter_controller_l((tmp, None, None, attn_output), task)
            
            del tmp
            
        elif "C" in self.lora and self.lora_head:
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            attn_output = self.c_proj(attn_output)
            _, attn_output = self.adapter_controller_l((attn_weights, key, None, attn_output), task)
            
            del attn_weights
            
        else:
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            attn_output = self.c_proj(attn_output)
        
        
        if self.prefix_tuning and self.prefix_value_extension:
            attn_output = attn_output + prefix_output


        if self.train_task_adapters:
            if self.parallel_adapter == "": 
                attn_output = self.adapter_controller(attn_output, task)
            else:
                attn_output = attn_output + self.adapter_controller(hidden_states, task)

        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)



class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, adapter_config=None):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.train_task_adapters = config.train_task_adapters and adapter_config.add_adapter_in_feed_forward
        if self.train_task_adapters:
            self.parallel_adapter = adapter_config.parallel_adapter
        
        self.train_kernel_adapter = config.kernel_adapter
        
        if self.train_task_adapters:
            adapter_config.reduction_factor = adapter_config.task_reduction_factor
            if self.train_kernel_adapter:
                adapter_config = copy.deepcopy(adapter_config)
                adapter_config.kernel_adapter = False
                adapter_config.n_head = config.n_head
            elif self.parallel_adapter:
                adapter_config = copy.deepcopy(adapter_config)
                adapter_config.kernel_adapter = False
                adapter_config.n_head = config.n_head
                adapter_config.lora = ""
                adapter_config.prefix_tuning = ""
            self.adapter_controller = AdapterController(adapter_config)

    def forward(self, hidden_states, task=None):
        if self.train_task_adapters and self.parallel_adapter: 
            para_output = self.adapter_controller(hidden_states, task)
        
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        if self.train_task_adapters:
            if self.parallel_adapter == "": 
                hidden_states = self.adapter_controller(hidden_states, task)
            else:
                hidden_states = hidden_states + para_output
                del para_output

        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, adapter_config=None,):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, adapter_config=adapter_config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config, adapter_config=adapter_config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        task=None
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            task=task
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, task=task)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of length :obj:`config.n_layers`, containing tuples of tensors of shape :obj:`(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            GPT2Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


GPT2_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0][0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.

            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.

            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48

    Example::

            # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
            model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
            device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8],

                          1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                          2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                          3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example::

        # On a 4 GPU machine with gpt2-large:
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7],

                    1: [8, 9, 10, 11, 12, 13, 14, 15],
                    2: [16, 17, 18, 19, 20, 21, 22, 23],
                    3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config, adapter_config=None):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        
        # self.h = nn.ModuleList([GPT2Block(self.per_layer_config(config, layer_id, adapter_config), adapter_config=adapter_config) for layer_id in range(config.num_hidden_layers)])
        self.h = nn.ModuleList([GPT2Block(*self.per_layer_config(config, layer_id, adapter_config)) for layer_id in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    ######################################################
    def per_layer_config(self, config, layer_id, adapter_config):
        """Sets the train_task_adapter in the config, based on the information given."""
        def is_valid_layer(layer_id, adapter_config):
            # valid_layer_ids = adapter_config.task_adapter_layers_decoder
            if valid_layer_ids is None:
                 return True
            return True if layer_id in valid_layer_ids else False
        
        def get_task_reduction_factor(layer_id, config, adapter_config):
            
            factor_list = adapter_config.task_reduction_factor_list
            
            if factor_list is None:
                # just use the task_reduction_factor
                return adapter_config.task_reduction_factor
            elif len(factor_list) == config.num_hidden_layers:
                # there is an adapter in each layer, while the settings for each layer are different
                return factor_list[layer_id]
            else:
                assert len(factor_list) == len(valid_layer_ids)
                return factor_list[valid_layer_ids.index(layer_id)]
                
        if adapter_config is None:
            return (config, adapter_config)
            
        config = copy.deepcopy(config)
        adapter_config = copy.deepcopy(adapter_config)
        
        valid_layer_ids = adapter_config.task_adapter_layers_decoder
        valid_task_adapter_layer_id = is_valid_layer(layer_id, adapter_config)
        config.train_task_adapters = config.train_task_adapters and\
                                     valid_task_adapter_layer_id
        
        if config.train_task_adapters:
            adapter_config.task_reduction_factor = get_task_reduction_factor(layer_id, config, adapter_config)
        
        if adapter_config.kernel_adapter and adapter_config.corresponding_kernel:
            adapter_config.kernel_adapter = config.train_task_adapters
            config.kernel_adapter = config.train_task_adapters
        
        # print("### config for layer ", layer_id, " is task", config.train_task_adapters)
        print("### config for layer ", layer_id, " is task", config.train_task_adapters, "reduction_factor", adapter_config.task_reduction_factor)
        return (config, adapter_config)
        #################################################### 


    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    task=task
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)

class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, adapter_config=None):
        super().__init__(config)
        self.transformer = GPT2Model(config, adapter_config=adapter_config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        ###########################################################################
        # setup the modules for prefix-tuning
        if adapter_config and adapter_config.prefix_tuning == "reparameterization":
        
            # self.prefix_tuning = adapter_config.prefix_tuning
            # self.preseqlen = adapter_config.prefix_length
            # self.input_tokens = torch.arange(self.preseqlen).long()
            # self.prefix_wte = nn.Embedding(self.preseqlen, config.n_embd)
            # self.prefix_control_trans = nn.Sequential(
                # nn.Linear(config.n_embd, adapter_config.prefix_dim),
                # nn.Tanh())
                
            self.prefix_tuning = adapter_config.prefix_tuning
            self.preseqlen = adapter_config.prefix_length
            self.mid_dim = adapter_config.prefix_dim
            
            print('[Full prefix-tuning Setting :) ]')
            self.input_tokens = torch.arange(self.preseqlen).long()
            self.prefix_wte = nn.Embedding(self.preseqlen, config.n_embd)
            self.prefix_control_trans = nn.Sequential(
                nn.Linear(config.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                
            
            self.match_n_layer = config.n_layer
            self.embed_dim = config.hidden_size
            self.match_n_head = config.num_attention_heads
            self.match_n_embd = self.embed_dim // self.match_n_head
            
        else:
            self.prefix_tuning = ""
            
        ###############################################################################
        ###########################################################################
        # Creates and sets a shared phm_rule in case of hypercomplex adapters with a shared phm_rule.
        if config.train_task_adapters and adapter_config.hypercomplex_adapters:
            
            # If shared_phm_rule is True: a common phm_rule is initiated across all layers
            if adapter_config.shared_phm_rule:
                phm_dim = adapter_config.hypercomplex_division
                self.factorized_phm_rule = adapter_config.factorized_phm_rule
                if self.factorized_phm_rule:
                    self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1).to(adapter_config.device),
                       requires_grad=adapter_config.learn_phm)
                    self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim).to(adapter_config.device),
                       requires_grad=adapter_config.learn_phm)
                    if adapter_config.phm_c_init == "normal":
                        self.phm_rule_left.data.normal_(mean=0, std=adapter_config.phm_init_range)
                        self.phm_rule_right.data.normal_(mean=0, std=adapter_config.phm_init_range)
                    elif adapter_config.phm_c_init == "uniform":
                        self.phm_rule_left.data.uniform_(-1, 1)
                        self.phm_rule_right.data.uniform_(-1, 1)
                    else:
                        raise NotImplementedError
                else:
                    self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim).to(adapter_config.device),\
                       requires_grad=adapter_config.learn_phm)
                    if adapter_config.phm_c_init == "normal":
                       self.phm_rule.data.normal_(mean=0, std=adapter_config.phm_init_range)
                    elif adapter_config.phm_c_init == "uniform":
                       self.phm_rule.data.uniform_(-1, 1)
                    else:
                       raise NotImplementedError 
                self.set_phm_rule()

            # TODO: clean this up later.
            if adapter_config.shared_W_phm:
                self.w_init=adapter_config.hypercomplex_nonlinearity
                self.phm_dim = adapter_config.hypercomplex_division
                down_sample_size = adapter_config.input_dim // adapter_config.task_reduction_factor
                in_feats_per_axis = adapter_config.input_dim // self.phm_dim
                out_feats_per_axis = down_sample_size // self.phm_dim
                self.factorized_phm = adapter_config.factorized_phm 
                if self.factorized_phm:
                    self.phm_rank = adapter_config.phm_rank 
                    self.W_down_left = nn.Parameter(torch.Tensor(size=(self.phm_dim, in_feats_per_axis, self.phm_rank)),
                              requires_grad=True)
                    self.W_down_right = nn.Parameter(torch.Tensor(size=(self.phm_dim, self.phm_rank, out_feats_per_axis)),
                              requires_grad=True)
                    self.W_up_left = nn.Parameter(torch.Tensor(size=(self.phm_dim, out_feats_per_axis, self.phm_rank)),
                              requires_grad=True)
                    self.W_up_right = nn.Parameter(torch.Tensor(size=(self.phm_dim, self.phm_rank, in_feats_per_axis)),
                              requires_grad=True)
                    self.init_W(in_feats_per_axis, out_feats_per_axis, W_left=self.W_down_left, 
                              W_right=self.W_down_right)
                    self.init_W(out_feats_per_axis, in_feats_per_axis, W_left=self.W_up_left, 
                              W_right=self.W_up_right)
                else:
                    self.W_down = nn.Parameter(torch.Tensor(size=(self.phm_dim, in_feats_per_axis, out_feats_per_axis)),
                              requires_grad=True)
                    self.W_up = nn.Parameter(torch.Tensor(size=(self.phm_dim, out_feats_per_axis, in_feats_per_axis)),
                              requires_grad=True)
                    self.init_W(in_feats_per_axis, out_feats_per_axis, W=self.W_down)
                    self.init_W(out_feats_per_axis, in_feats_per_axis, W=self.W_up)
                self.set_phm_Ws()
        ###############################################################################

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    
    ###############################################
    
    def set_phm_Ws(self):
      def set_phm_Ws_helper(module): 
        # TODO: we need to check there is one of these, and this is activated.
        for name, sub_module in module.named_modules():
            if isinstance(sub_module, PHMLinear) and "down_sampler" in name:
                if self.factorized_phm:
                    sub_module.set_W(W_left=self.W_down_left, W_right=self.W_down_right)
                else:
                    sub_module.set_W(W=self.W_down)
            if isinstance(sub_module, PHMLinear) and "up_sampler" in name:
                if self.factorized_phm:
                    sub_module.set_W(W_left=self.W_up_left, W_right=self.W_up_right)
                else:
                    sub_module.set_W(W=self.W_up)

      set_phm_Ws_helper(self.transformer)  

    def init_W(self, in_feats_per_axis, out_feats_per_axis, W=None, W_left=None, W_right=None):
        if self.w_init == "glorot-normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    W_left.data[i] = glorot_normal(W_left.data[i])
                    W_right.data[i] = glorot_normal(W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    W.data[i] = glorot_normal(W.data[i])
        elif self.w_init == "glorot-uniform":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    W_left.data[i] = glorot_uniform(W_left.data[i])
                    W_right.data[i] = glorot_uniform(W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    W.data[i] = glorot_uniform(W.data[i])
        elif self.w_init == "normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    W_left.data[i].normal_(std=0.01)
                    W_right.data[i].normal_(std=0.01)
            else:
                for i in range(self.phm_dim):
                    W.data[i].normal_(std=0.01)
        else:
            raise ValueError

    def set_phm_rule(self):
        def set_phm_rule(module):
            # TODO: we need to check there is one of these, and this is activated.
            for name, sub_module in module.named_modules():
                if isinstance(sub_module, PHMLinear):
                    if self.factorized_phm_rule:
                        sub_module.set_phm_rule(phm_rule_right=self.phm_rule_right, 
                                                phm_rule_left=self.phm_rule_left)
                    else:
                        sub_module.set_phm_rule(phm_rule=self.phm_rule)
        set_phm_rule(self.transformer)
    ###########################################################


    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
    
    def get_prompt_p5(self, bsz=None):
        # temp_control = self.prefix_wte(self.input_tokens.to(self.device))
        # prefix_mid_embd = self.prefix_control_trans(temp_control) # prefix_len, mid_dim
        
        # return prefix_mid_embd

        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.prefix_wte(input_tokens)
        past_key_values = self.prefix_control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smooth=0.0,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        # _batch, _len = input_ids.shape[0], input_ids.shape[-1]-1
        # print(input_ids.shape) # batch,1,271
        
        #################################################
        if self.prefix_tuning == "reparameterization":
            prefix_mid_embd = self.get_prompt_p5(bsz=input_ids.shape[0])
            past_key_values = prefix_mid_embd
        else:
            prefix_mid_embd = None
        #################################################
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        
        
        loss = None
        if labels is not None:
            """
            if label_smooth > 0.0001:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                lm_logits = shift_logits
                lm_labels = shift_labels
            
                logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                # loss = loss.view(_batch, _len)
                
                
                loss = loss.mean()
                
            else:
                # loss in LoRA with 0 label_smooth
                # !!! reduce=False / shift_labels?
                #### check shift labels / logits
                # loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)
            """
            # loss in our code
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous() # remove the last logit
            shift_labels = labels[..., 1:].contiguous() # remove the first label
            # Flatten the tokens
            if label_smooth > 1e-6:
                loss_fct = CrossEntropyLoss(label_smoothing=label_smooth)
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # the final loss to return should be of shape torch.Size([]), a scalar

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


