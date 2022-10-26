"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
import torch.nn as nn
from .adapter_utils import Activations
from hypercomplex.layers import PHMLinear
from .low_rank_layer import LowRankLinear, LoRALinear, PrefixMLP
from .low_rank_layer import truncated_gaussian, truncated_gaussian_initialize, initial_val_plus_disturbance
import torch
from torch.nn.parameter import Parameter

class KernelAdapter(nn.Module):
    """This is the proposed Kernel Adapter, in which each attention matrix of (Q,K) are scaled using a bandwith paramters
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.n_head = config.n_head

        if config.kerfit_variant_type == "kerfit_X": # applied to X
            shape = (self.input_dim, )
        else: # applied to Q
            if config.shared_bandwidth:
                shape = (self.input_dim//self.n_head, )
            else:
                shape = (self.n_head, 1, self.input_dim//self.n_head)        
        
        
        if config.kerfit_variant_type in ["", "kernel", "bitfit"]:
            
            if config.kerfit_variant_type == "kernel":
                self.config.kernel_bitfit_variant = False
            elif config.kerfit_variant_type == "bitfit":
                self.config.kernel_bitfit_variant = True
        
            if self.config.kernel_bitfit_variant:
                initialize_func = torch.zeros 
            else:
                initialize_func = torch.ones
            
            self.query_bandwidth = Parameter(
                initial_val_plus_disturbance(initialize_func, shape, 
                    config.truncated_gaussian_initialization))
            
            if not config.kernel_Q_only:
                self.key_bandwidth = Parameter(
                    initial_val_plus_disturbance(initialize_func, shape, 
                        config.truncated_gaussian_initialization))
        
        elif config.kerfit_variant_type in ["kerfit", "kerfit_sep", "kerfit_X"]: # must be Q only
            self.query_bias = Parameter(initial_val_plus_disturbance(torch.zeros, shape, 
                config.truncated_gaussian_initialization)) # Initializing with 0s
            
            self.query_bandwidth = Parameter(
                initial_val_plus_disturbance(torch.ones, shape, 
                config.truncated_gaussian_initialization)) # Initializing with 1s
                
            if not config.kernel_Q_only:
                self.key_bandwidth = Parameter(
                    initial_val_plus_disturbance(torch.ones, shape, 
                    config.truncated_gaussian_initialization)) # Initializing with 1s
            
            if config.kerfit_variant_type == "kerfit_sep":
                if config.shared_bandwidth:
                    shape = torch.Size([])
                else:
                    shape = (self.n_head,1,1)
                
                self.scaling_factor = Parameter(initial_val_plus_disturbance(torch.ones, shape, 
                    config.truncated_gaussian_initialization))
        
        else:
            raise ValueError

    def forward(self, x):
        # Scaling the matrix by the bandwith parameter
        # Note: Shape of query and key matrices are - (batch, num_of_heads, seq_len, hidden_dimension). Here hidden_dim = 768/num_of_heads = 768/12 = 64
        query, key = x
        # query = query/self.query_bandwidth 
        # key = key/self.key_bandwidth
        # 12, 1, hidden
        # key = query
        # K(q/b, k/b)
        if self.config.kerfit_variant_type in ["kerfit", "kerfit_sep", "kerfit_X"]:
            if self.config.kerfit_variant_type == "kerfit_sep":
                scale = self.query_bandwidth * self.scaling_factor
            # elif self.config.kerfit_variant_type in ["kerfit_X", "kerfit"]:
            elif self.config.kerfit_variant_type in []:
                scale = 1 + (self.query_bandwidth - 1) * 4
            # attempt failed before, no longer in use
            else:
                scale = self.query_bandwidth
            query = (query + self.query_bias) * scale
            if not self.config.kernel_Q_only:
                key = key * self.key_bandwidth

            
        else:
            if self.config.kernel_bitfit_variant:
                query = query + self.query_bandwidth
                if not self.config.kernel_Q_only:
                    key = key + self.key_bandwidth
            else:
                query = query * self.query_bandwidth
                if not self.config.kernel_Q_only:
                    key = key * self.key_bandwidth
        
        return (query, key)

class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices. c.f. LowRankLinear()
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = int(self.input_dim // config.reduction_factor)
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
                                          w_init=config.low_rank_w_init,
                                          rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
                                        w_init=config.low_rank_w_init,
                                        rank=config.low_rank_rank)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = int(self.input_dim // config.reduction_factor)
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        
        if config.truncated_gaussian_initialization:
            if config.initialization_scale == "ntk":
                scale1 = 0.01
                scale2 = 0.1 * (self.down_sample_size / 2)**(-0.5)
            else:
                scale1 = float(config.initialization_scale)
                scale2 = scale1
            truncated_gaussian_initialize(self.down_sampler, scale1)
            truncated_gaussian_initialize(self.up_sampler, scale2)
        
        

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output

class LoRA(nn.Module):
    """Restrict the difference between pre-trained V weights and trained V weights to be low-rank."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.q_r = config.lora_Q_rank
        self.v_r = config.lora_V_rank
        
        self.q_share_r = config.lora_Q_share_rank
        self.v_share_r = config.lora_V_share_rank
        
        if self.config.lora_head:
            self.output_dim = self.input_dim // config.n_head
        else:
            self.output_dim = self.input_dim
        
        if "Q" in self.config.lora:
            q_r = self.q_r if config.kerfit_variant_type == "" else self.q_r-1
            q_bias = config.kerfit_variant_type == ""
            self.WQ_LoRA = LoRALinear(self.input_dim, self.output_dim, q_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                # self.config.lora_ini, self.config.lora_head, self.config.lora_head_base,
                self.config.lora_ini, self.config.lora_head, self.config.lora_head_base[0],
                q_bias)
            
            if self.q_share_r > 0:
                self.WQ_share_LoRA = LoRALinear(self.input_dim, self.input_dim, self.q_share_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_ini, lora_head=False, lora_head_base="X", bias=False)
                
        if "K" in self.config.lora:
            q_r = self.q_r if config.kerfit_variant_type == "" else self.q_r-1
            q_bias = False
            self.WQ_LoRA = LoRALinear(self.input_dim, self.output_dim, q_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_ini, self.config.lora_head, self.config.lora_head_base[0],
                q_bias)
        
        if "V" in self.config.lora:
            # base = self.config.lora_head_base
            # if self.config.lora_head_base == "QS":
                # base = "K"
            
            self.WV_LoRA = LoRALinear(self.input_dim, self.output_dim, self.v_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_ini, self.config.lora_head, self.config.lora_head_base[1])
                
            if self.v_share_r > 0:
                self.WV_share_LoRA = LoRALinear(self.input_dim, self.input_dim, self.v_share_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_ini, lora_head=False, lora_head_base="X", bias=False)      
        
        if "O" in self.config.lora:
            # 64 x 768
            self.WO_LoRA = LoRALinear(self.output_dim, self.input_dim, self.v_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_ini, self.config.lora_head, self.config.lora_head_base[-1])
            
            if self.v_share_r > 0:
                self.WO_share_LoRA = LoRALinear(self.input_dim, self.input_dim, self.v_share_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_ini, lora_head=False, lora_head_base="A", bias=False)     
        
        # "C" for combination of W_V and W_O. Update (W_V^i W_O^i) as a whole
        if "C" in self.config.lora:
            # 64 x 768
            self.WC_LoRA = LoraAdapter(self.config)

        
        self.bases = {"X": 0, "K": 1, "Q": 2, "V": 3, "A": 0}
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        
    def forward(self, x):
        X, key, query, value = x
        # if lora_head = true, Q.shape = (batch, head, seq_length, head_features)
                
        
        if "V" in self.config.lora and query is not None:
            base = x[self.bases[self.config.lora_head_base[1]]]
            value = value + self.WV_LoRA(base)
            
            if self.v_share_r > 0:
                value = value + self._split_heads(self.WV_share_LoRA(X), value.shape[1], value.shape[3])
            
        if "Q" in self.config.lora and query is not None:
            # if self.config.lora_head_base == "QS": X = X - query
            base = x[self.bases[self.config.lora_head_base[0]]]
            query = query + self.WQ_LoRA(base)
            
            if self.q_share_r > 0:
                query = query + self._split_heads(self.WQ_share_LoRA(X), query.shape[1], query.shape[3])
            
        if "K" in self.config.lora and key is not None:
            # if self.config.lora_head_base == "QS": X = X - query
            base = x[self.bases[self.config.lora_head_base[0]]]
            query = key + self.WQ_LoRA(base)
        
        if "O" in self.config.lora and query is None:
            
            base = x[self.bases[self.config.lora_head_base[-1]]]
            value = value + self.WO_LoRA(base)
            
            if self.v_share_r > 0:
                value = value + self.WO_share_LoRA(X)
        
        if "C" in self.config.lora and query is None:
            # value = attn_out, X = attn_weights
            base = x[self.bases[self.config.lora_head_base[-1]]]
            value = value + self.WC_LoRA((X, base))
        
        return (query, value)

class LoraAdapter(nn.Module):
    """
    This is the adapter for LoRA VO. Failed before, no longer in use.
    """
    def __init__(self, config, 
        lora_tanh: bool = True, lora_ini: str = "Houlsby", 
        bias: bool = True):
        
        super().__init__()
        self.config = config
        rank = config.lora_V_rank
        input_dim = config.input_dim # 768
        output_dim = input_dim // config.n_head # 64
        
        self.lora_head_base = config.lora_head_base[-1]
        self.bias = bias
        
        if lora_tanh:
            self.lora_tanh = torch.tanh
        else:
            self.lora_tanh = lambda x: x
        
        if rank > 0:
            
            # if lora_head = true, Q.shape = (batch, head, seq_length, head_features)
            if self.lora_head_base in ["Q", "K"]:
                self.lora_A = nn.Parameter(torch.Tensor(size=(config.n_head, output_dim, rank)), requires_grad=True)
                self.lora_B = nn.Parameter(torch.Tensor(size=(config.n_head * rank, input_dim)), requires_grad=True)
                if bias:
                    self.A_bias = nn.Parameter(torch.Tensor(config.n_head, 1, rank))
                    self.B_bias = nn.Parameter(torch.Tensor(input_dim))
        
        self.reset_parameters(lora_ini)
    
    def reset_parameters(self, lora_ini):
        
        if lora_ini == "Houlsby":
            if self.bias: 
                self.A_bias.data = truncated_gaussian(self.A_bias.shape, 0.01)
                self.B_bias.data = truncated_gaussian(self.B_bias.shape, 0.01)
            
            self.lora_A.data = truncated_gaussian(self.lora_A.shape, 0.01)
            self.lora_B.data = truncated_gaussian(self.lora_B.shape, 0.01)   
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def forward(self, x):

        attn_weights, key = x
        
        value = torch.einsum("bhnd,hdr->bhnr", key, self.lora_A)
        if self.bias: value = value + self.A_bias
        value = self.lora_tanh(value)
        
        value = torch.matmul(attn_weights, value)
        value = self._merge_heads(value, self.config.n_head, value.shape[-1]) # bn(hr)
        
        value = torch.matmul(value, self.lora_B)
        if self.bias: value = value + self.B_bias

        return value
        
        
class PrefixAdapter(nn.Module):
    """This is the adapter for prefix-tuning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.mid_dim = config.prefix_dim
        self.output_dim = self.input_dim // config.n_head
        
        if (config.prefix_tuning == "landmark" or config.prefix_tuning == "MH_adapter" 
            or config.prefix_tuning == "value_gating"):
            
            self.prefix_v_mlp = PrefixMLP(self.output_dim, config.n_head, self.mid_dim, 
                                    prefix_value_extension=config.prefix_value_extension,
                                    prefix_head_share=config.prefix_head_share)
            
            if config.prefix_tuning == "value_gating": 
                self.prefix_v_gating_mlp = PrefixMLP(self.output_dim, config.n_head, self.mid_dim, 
                                    prefix_head_base = 'K',
                                    prefix_value_extension=config.prefix_value_extension,
                                    prefix_head_share=config.prefix_head_share)
            
            elif config.prefix_tuning == "landmark" and config.prefix_length >= 2:
                self.prefix_v_pert_mlp = PrefixMLP(self.output_dim, config.n_head, config.prefix_length, 
                                    prefix_value_extension=False,
                                    prefix_head_share=False)
                                    
    def forward(self, x):
        if self.config.prefix_tuning == "value_gating":
            if self.config.prefix_value_extension:
                query, key, prefix_weights, attn_weights = x
                prefix_v = self.prefix_v_mlp((query, prefix_weights))
                prefix_v_gating = self.prefix_v_gating_mlp((key, attn_weights))
                prefix_v = (prefix_v, prefix_v_gating)
            else:
                query, key = x
                prefix_v = self.prefix_v_mlp(query)
                prefix_v_gating = self.prefix_v_gating_mlp(key)
                prefix_v = (prefix_v, prefix_v_gating)
        elif self.config.prefix_tuning == "landmark" and type(x) == tuple and x[-1] is None:
            prefix_v = self.prefix_v_pert_mlp(x[0])
        else:
            prefix_v = self.prefix_v_mlp(x)

        return prefix_v



class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = int(self.input_dim // config.reduction_factor)
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=config.phm_c_init,
                                      phm_dim=config.hypercomplex_division,
                                      learn_phm=config.learn_phm,
                                      w_init=config.hypercomplex_nonlinearity,
                                      shared_phm_rule=config.shared_phm_rule,
                                      factorized_phm=config.factorized_phm,
                                      shared_W_phm=config.shared_W_phm,
                                      factorized_phm_rule=config.factorized_phm_rule,
                                      phm_rank=config.phm_rank,
                                      phm_init_range=config.phm_init_range,
                                      kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.input_dim, 
                                    bias=True,
                                    c_init=config.phm_c_init,
                                    phm_dim=config.hypercomplex_division,
                                    learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm,
                                    shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank,
                                    phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)
    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)