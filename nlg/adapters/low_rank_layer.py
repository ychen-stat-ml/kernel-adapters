"""This script implements a low-rank linear layer."""
import torch 
import torch.nn as nn
import torch.nn.functional as F

from hypercomplex.inits import glorot_uniform, glorot_normal

def truncated_gaussian(shape, std):
    return torch.fmod(torch.randn(shape), 2) * std

def truncated_gaussian_initialize(module, std):
    for param in module.parameters():
        new_weights = truncated_gaussian(param.shape, std)
        
        with torch.no_grad():
            param.copy_(new_weights)

def initial_val_plus_disturbance(initialize_func, shape, ini_flag, std=0.01):
    
    val = initialize_func(shape)
    disturbance_q = truncated_gaussian(val.shape, std) if ini_flag else 0
    return val + disturbance_q



class LowRankLinear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 1,
        bias: bool = True, w_init: str = "glorot-uniform"):
        super(LowRankLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.rank = rank
        self.bias = bias
        self.w_init = w_init
        self.W_left = nn.Parameter(torch.Tensor(size=(input_dim, rank)), requires_grad=True)
        self.W_right = nn.Parameter(torch.Tensor(size=(rank, output_dim)), requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
        if self.w_init == "glorot-uniform": 
            self.W_left.data = glorot_uniform(self.W_left.data) 
            self.W_right.data = glorot_uniform(self.W_right.data)          
        elif self.w_init == "glorot-normal":
            self.W_left.data = glorot_normal(self.W_left.data)
            self.W_right.data = glorot_normal(self.W_right.data)
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W_left*self.W_right
        output = torch.matmul(input=x, other=W)
        if self.bias:
            output += self.b
        return output

# follow the settings in LoRA code https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class LoRALinear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 0, 
        lora_alpha: float = 1.0, lora_dropout: float = 0.0, lora_tanh: bool = False, 
        lora_ini: str = "He_s", lora_head: bool = False, 
        lora_head_base: str = "X", bias: bool = True):
        
        super(LoRALinear, self).__init__()
        in_features = input_dim
        out_features = output_dim 
        r = rank
        # print(r)
        
        self.lora_head = lora_head
        self.lora_head_base = lora_head_base
        self.bias = bias
        
        
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        if lora_tanh:
            self.lora_tanh = torch.tanh
        else:
            self.lora_tanh = lambda x: x
        
        if r > 0:
            self.scaling = 1
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            # self.lora_dropout = nn.Dropout(p=0.1)
            
            if not self.lora_head:
                if lora_alpha > 1.5: self.scaling = lora_alpha / r
                self.lora_A = nn.Parameter(torch.Tensor(size=(r, in_features)), requires_grad=True)
                self.lora_B = nn.Parameter(torch.Tensor(size=(out_features, r)), requires_grad=True)
                if bias:
                    self.b = nn.Parameter(torch.Tensor(output_dim))
            
            # if lora_head = true, Q.shape = (batch, head, seq_length, head_features)
            else:
                n_head = max(in_features // out_features, 
                            out_features // in_features)
                self.n_head = n_head
                
                if self.lora_head_base == "X":
                    self.lora_A = nn.Parameter(torch.Tensor(size=(n_head, r, in_features)), requires_grad=True)
                    self.lora_B = nn.Parameter(torch.Tensor(size=(n_head, out_features, r)), requires_grad=True)
                    if bias:
                        self.b = nn.Parameter(torch.Tensor(n_head, 1, out_features))
                elif self.lora_head_base in ["K", "Q"]:
                    self.lora_A = nn.Parameter(torch.Tensor(size=(n_head, r, out_features)), requires_grad=True)
                    self.lora_B = nn.Parameter(torch.Tensor(size=(n_head, out_features, r)), requires_grad=True)
                    if bias:
                        self.b = nn.Parameter(torch.Tensor(n_head, 1, out_features))
                elif self.lora_head_base == "A":
                    self.lora_A = nn.Parameter(torch.Tensor(size=(n_head, r, in_features)), requires_grad=True)
                    self.lora_B = nn.Parameter(torch.Tensor(size=(out_features, n_head * r)), requires_grad=True)
                    if bias:
                        self.b = nn.Parameter(torch.Tensor(output_dim))
                # elif self.lora_head_base == "QS":
                    # n_head = in_features // out_features

                    # self.lora_S = nn.Parameter(torch.Tensor(size=(n_head, out_features, 2*r)), requires_grad=True)
                    # if bias:
                        # self.b = nn.Parameter(torch.Tensor(n_head, 1, out_features))
        
        # if not lora_head: lora_ini = "He"
        self.reset_parameters(lora_ini)
    
    def reset_parameters(self, lora_ini):
        
        if lora_ini == "Houlsby":
            if self.bias: self.b.data = truncated_gaussian(self.b.shape, 0.01)
            # if self.lora_head_base == "QS":       
                # self.lora_S.data = truncated_gaussian(self.lora_S.shape, 0.01)
            # else:
            self.lora_A.data = truncated_gaussian(self.lora_A.shape, 0.01)
            self.lora_B.data = truncated_gaussian(self.lora_B.shape, 0.01)            
        else:
            if lora_ini == "He_s":
                a = 400*5**0.5
            elif lora_ini == "He":
                a = 5**0.5
            else:
                raise NameError
            if self.bias: self.b.data = torch.zeros_like(self.b.data)
            nn.init.kaiming_uniform_(self.lora_A, a=a)
            nn.init.zeros_(self.lora_B)

        """
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
            
        if self.lora_head:
            a = 400*5**0.5
        else:
            a = 5**0.5
        
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=a)
            nn.init.zeros_(self.lora_B)
            
        if self.lora_head_base != "QS":  
            delta_W = self.lora_tanh(self.lora_B @ self.lora_A * self.scaling)
        else:
            delta_W = torch.einsum("hpr,hdr->hpd", self.lora_S, self.lora_S)
            delta_W = self.lora_tanh(delta_W * self.scaling)    
            
        """
        
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        delta_W = self.lora_tanh(self.lora_B @ self.lora_A * self.scaling)
            
        if self.lora_head:
            if self.lora_head_base == "X": # d = 768, p = 64
                output = torch.einsum("bnd,hpd  ->bhnp", self.lora_dropout(x), delta_W)
            elif self.lora_head_base in ["Q", "K", "V", "QS"]: # d, p = 64
                x = self.lora_dropout(x)
                output = torch.einsum("bhnd,hpd  ->bhnp", x, delta_W)
                # if self.lora_head_base == "QS":
                    # output = output - 1e-4 * self.lora_S.size(-1) * x
            elif self.lora_head_base == "A": # d = 64, p = 768
                x = self.lora_dropout(x)
                output = torch.einsum("bhnd,hpd  ->bhnp", x, delta_W)
                output = output.sum(1)
        else:
            output = torch.einsum("bnd,pd->bnp", self.lora_dropout(x), delta_W)
        """
        if self.lora_head:
            if self.lora_head_base == "X": # d = 768, p = 64
                output = torch.einsum("bnd,hrd->bhnr", self.lora_dropout(x), self.lora_A * self.scaling)
                output = torch.einsum("bhnr,hpr->bhnp", output, self.lora_B)
                
            elif self.lora_head_base in ["Q", "K", "V", "QS"]: # d, p = 64
                output = torch.einsum("bhnd,hrd->bhnr", self.lora_dropout(x), self.lora_A * self.scaling)
                output = torch.einsum("bhnr,hdr->bhnd", output, self.lora_B)
            
            # change the assumption: x is multihead -> x is after merging
            elif self.lora_head_base == "A": # d = 64, p = 768
                output = torch.einsum("bhnd,hrd->bhnr", self.lora_dropout(x), self.lora_A * self.scaling)
                output = self._merge_heads(output, output.shape[1], output.shape[3]) # bn(hr)
                output = torch.einsum("bnR,dR->bnd", output, self.lora_B)
                
        else:
            if self.lora_head_base == "A":
                x = self._merge_heads(x, x.shape[1], x.shape[3])
                
            output = torch.einsum("bnp,rp->bnr", self.lora_dropout(x), self.lora_A * self.scaling)
            output = torch.einsum("bnr,pr->bnp", output, self.lora_B)
        
        if self.bias:
            output = output + self.b * self.scaling
        return output

class PrefixMLP(torch.nn.Module):
    def __init__(self, input_dim: int, n_head: int, rank: int = 0, 
        prefix_tanh: bool = True, prefix_ini: str = "Houlsby", 
        prefix_head_base: str = "Q", bias: bool = True, 
        prefix_value_extension: bool = False, prefix_head_share: bool = False):
        
        super(PrefixMLP, self).__init__()
        
        self.prefix_head_base = prefix_head_base
        self.bias = bias
        self.prefix_value_extension = prefix_value_extension
        self.prefix_head_share = prefix_head_share
        
        if prefix_tanh:
            self.prefix_tanh = torch.tanh
        else:
            self.prefix_tanh = lambda x: x
        
        if rank > 0:
            
            # if prefix_head = true, Q.shape = (batch, head, seq_length, head_features)
            if self.prefix_head_base in ["K", "Q"]:
                if self.prefix_value_extension:
                    self.prefix_A = nn.Parameter(torch.Tensor(size=(n_head, input_dim, rank)), requires_grad=True)
                    if bias: 
                        self.A_bias = nn.Parameter(torch.Tensor(n_head, 1, rank))
                        self.B_bias = nn.Parameter(torch.Tensor(n_head * input_dim))
                    
                    if self.prefix_head_share:
                        self.prefix_B = nn.Parameter(torch.Tensor(size=(rank, n_head * input_dim)), requires_grad=True)
                    else:
                        self.prefix_B = nn.Parameter(torch.Tensor(size=(n_head, rank, n_head * input_dim)), requires_grad=True)

                    
                else:
                    self.prefix_A = nn.Parameter(torch.Tensor(size=(n_head, input_dim, rank)), requires_grad=True)
                    self.prefix_B = nn.Parameter(torch.Tensor(size=(n_head, rank, input_dim)), requires_grad=True)
                
                    if bias:
                        self.A_bias = nn.Parameter(torch.Tensor(n_head, 1, rank))
                        self.B_bias = nn.Parameter(torch.Tensor(n_head, 1, input_dim))
        
        self.reset_parameters(prefix_ini)
    
    def reset_parameters(self, prefix_ini):
        
        if prefix_ini == "Houlsby":
            if self.bias: 
                self.A_bias.data = truncated_gaussian(self.A_bias.shape, 0.01)
                self.B_bias.data = truncated_gaussian(self.B_bias.shape, 0.01)
            
            self.prefix_A.data = truncated_gaussian(self.prefix_A.shape, 0.01)
            self.prefix_B.data = truncated_gaussian(self.prefix_B.shape, 0.01)            
            
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.prefix_value_extension:
            x, prefix_weights = x
        
        if self.prefix_head_base in ["K", "Q"]: # d, p = 64
            x = torch.einsum("bhnd,hdr->bhnr", x, self.prefix_A)
            if self.bias: x = x + self.A_bias
            x = self.prefix_tanh(x)
            
            if self.prefix_value_extension: # return bnp
                if prefix_weights is not None:                    
                    if self.prefix_head_base == "Q":
                        x = torch.einsum("bhn,bhnr->bhnr", prefix_weights, x)
                    else:
                        x = torch.matmul(prefix_weights, x) # "bhmn,bhnr->bhmr"
                    
                if self.prefix_head_share:
                    x = torch.einsum("bhnr,rp->bnp", x, self.prefix_B)
                else:
                    x = torch.einsum("bhnr,hrp->bnp", x, self.prefix_B)
                    
            else: # return bhnd
                x = torch.einsum("bhnr,hrd->bhnd", x, self.prefix_B)
                
            if self.bias: x = x + self.B_bias
            
        return x

