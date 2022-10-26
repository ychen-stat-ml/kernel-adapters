"""Implements the adapters and other parameter-efficient finetuning methods' configurations."""

from collections import OrderedDict
from dataclasses import dataclass

import torch.nn as nn

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751.
    We additionally pass all the configuration of parameter-efficient finetuning
    methods with this config."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    task_reduction_factor: int = 16
    task_reduction_factor_list = None
    add_adapter_in_feed_forward = True
    add_adapter_in_self_attention = True
    hidden_dim = 128
    task_adapter_layers_decoder = None
    
    parallel_adapter = ""

    # Hypercomplex adapters parameters 
    hypercomplex_adapters = False
    hypercomplex_division = 8
    learn_phm = True
    hypercomplex_nonlinearity="glorot-uniform"
    shared_phm_rule = False 
    factorized_phm = False 
    shared_W_phm = False
    factorized_phm_rule = False 
    phm_c_init = "normal"
    phm_rank = 1
    phm_init_range=0.01

    # prefix-tuning parameters.
    prefix_dim = 100
    init_prefix_from_vocab = False 
    kronecker_prod = False  

    # BitFit configuration.
    bitfit = False

    # Low-rank adapters.
    low_rank_adapters = False
    low_rank_w_init = "glorot-uniform"
    low_rank_rank = 1

    # Kernel adapter
    kernel_adapter = False
    shared_bandwidth = True
    kernel_bitfit_variant = False
    corresponding_kernel = False
    kernel_Q_only = False
    kerfit_variant_type = ""
    
    lora = ""
    lora_Q_rank = 4
    lora_V_rank = 4
    lora_alpha = 1.0
    lora_dropout = 0.0
    lora_tanh = False
    lora_ini = "He_s"
    lora_head = False
    lora_head_base = "XX"
    lora_Q_share_rank = 0
    lora_V_share_rank = 0
    
    truncated_gaussian_initialization = False
    initialization_scale = "0.01"
    
    prefix_tuning = ""
    prefix_length = 8
    prefix_dim = 512
    prefix_value_extension = False
    prefix_head_share = False
    prefix_gating_ablation = False
    
    init_prefix_from_vocab = False



ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig)])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
                .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))
