import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import RobertaConfig

def init_lisa_params(module):
    std = 1e-20
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def init_bias_mlp(module):
    std = 1e-2
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def init_zero_weights(module):
    if isinstance(module, nn.Embedding):
        nn.init.constant_(module.weight, 0.0)


class Prefix(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte_enc = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  },
                         }

            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                            }
            key_val_enc = past_key_values_enc[i]
            # at generation time, this is expanded automatically to the beam size
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_value": key_val_enc[1].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device)
                                    }
            result.append(temp_dict)
        return result


class PrefixCrossAttn(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'encoder_decoder': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  },
                         }
            key_val2 = past_key_values2[i]
            temp_dict['self'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                        "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                        "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                        }
            result.append(temp_dict)
        return result


class PrefixDirectInit(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)
        self.input_tokens = torch.arange(self.attn_bn).long()
        self.encoder_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                for _ in range(self.match_n_layer)])
        self.encoder_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                               for _ in range(self.match_n_layer)])
        self.decoder_self_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                     for _ in range(self.match_n_layer)])
        self.decoder_self_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                    for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_key = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                      for _ in range(self.match_n_layer)])
        self.decoder_cross_attn_value = nn.ModuleList([nn.Embedding(self.attn_bn, self.n_embd)
                                                     for _ in range(self.match_n_layer)])

        # fixme: choose a favorable init method
        self.apply(init_bert_weights)

    def _shape(self, x, bsz):
        y = x.view(bsz, self.attn_bn, self.match_n_head, self.match_n_embd)
        y = y.permute([0, 2, 1, 3])
        y = y.contiguous().view(bsz * self.match_n_head, -1, self.match_n_embd)
        return y

    def forward(self, bsz, nsamples=1, device="cuda"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)

        result = []
        for i, (enc_attn_k, enc_attn_v, dec_self_attn_k, dec_self_attn_v, dec_xattn_k, dec_xattn_v) in \
                enumerate(zip(self.encoder_attn_key, self.encoder_attn_value, self.decoder_self_attn_key,
                              self.decoder_self_attn_value, self.decoder_cross_attn_key, self.decoder_cross_attn_value)):
            temp_dict = {'self': {"prev_key": self._shape(dec_self_attn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_self_attn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.attn_bn).to(device) #bsz, attn_bn
                                  },
                         'encoder_decoder': {"prev_key": self._shape(dec_xattn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_xattn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.attn_bn).to(device)  #bsz, attn_bn
                                  },
                         'encoder': {"prev_key": self._shape(enc_attn_k(input_tokens_enc), old_bsz),
                                  "prev_value": self._shape(enc_attn_v(input_tokens_enc), old_bsz),
                                  "prev_key_padding_mask": torch.zeros(old_bsz, self.attn_bn).to(device) #bsz, attn_bn
                                  },
                        }
            result.append(temp_dict)
        return result


class MLP_Bias(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.attn_bn = args.attn_bn
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.src_len = config.max_source_length + 2
        self.tgt_len = config.max_target_length + 2
        self.tgt_input_tokens = torch.arange(self.tgt_len).long()
        self.src_input_tokens = torch.arange(self.src_len).long()

        self.wte = nn.Embedding(self.tgt_len, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.wte_enc = nn.Embedding(self.src_len, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.wte2 = nn.Embedding(self.tgt_len, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.apply(init_bias_mlp)
        # # initialization
        # nn.init.constant_(self.wte.weight, 0.0)
        # nn.init.constant_(self.wte_enc.weight, 0.0)
        # nn.init.constant_(self.wte2.weight, 0.0)

    def forward(self, bsz, nsamples=1, device="cuda"):
        temp_control = self.wte(self.tgt_input_tokens.to(device))
        past_key_values = self.control_trans(temp_control)  # tgt_len, layer*emb
        past_key_values = past_key_values.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values = self.dropout(past_key_values)

        temp_control2 = self.wte2(self.tgt_input_tokens.to(device))
        past_key_values2 = self.control_trans2(temp_control2)  # tgt_len, layer*emb
        past_key_values2 = past_key_values2.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values2 = self.dropout(past_key_values2)

        temp_control_enc = self.wte_enc(self.src_input_tokens.to(device))
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # src_len, layer*emb
        past_key_values_enc = past_key_values_enc.view(self.src_len, self.match_n_layer, self.n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)

        result = []
        for ii in range(self.match_n_layer):
            temp_dict = {"encoder": past_key_values_enc[:, ii, :],
                         "self": past_key_values[:, ii, :],
                         "encoder_decoder": past_key_values2[:, ii, :]}
            result.append(temp_dict)
        return result


class Bias(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.match_n_layer = config.num_hidden_layers
        self.n_embd = config.d_model

        # Option 1: a simple version, no transformations, each attention layer has its own bias parameters
        self.encoder_attn_bias = nn.ModuleList([nn.Embedding(args.max_source_length + 2, self.n_embd)
                                                for _ in range(self.match_n_layer)])
        self.decoder_self_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                     for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                      for _ in range(self.match_n_layer)])
        for embed in self.encoder_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)
        for embed in self.decoder_self_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)
        for embed in self.decoder_cross_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)

    def forward(self, bsz, nsamples=1, device="cuda"):
        result = []
        max_src_len = self.args.max_source_length + 2
        max_tgt_len = self.args.max_target_length + 2

        src_positions = torch.arange(0, max_src_len, dtype=torch.long, device=device)
        tgt_positions = torch.arange(0, max_tgt_len, dtype=torch.long, device=device)
        for ii in range(self.match_n_layer):
            temp_dict = {"encoder": self.encoder_attn_bias[ii].forward(src_positions),
                         "self": self.decoder_self_attn_bias[ii].forward(tgt_positions),
                         "encoder_decoder": self.decoder_cross_attn_bias[ii].forward(tgt_positions)}
            result.append(temp_dict)
        return


class Adapter_Layer(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        # self.non_linearity = args.non_linearity  # use ReLU by default

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


def softmax_gating(logits_1, logits_2):
    # the last two dimensions of logits is (T, S)
    max_logits = torch.max(torch.cat([logits_1, logits_2], dim=-1), dim=-1, keepdim=True)[0]
    logits_1 = logits_1 - max_logits
    logits_2 = logits_2 - max_logits
    exp_logits_1 = logits_1.exp()
    exp_logits_2 = logits_2.exp()
    s = torch.sum(exp_logits_1, dim=-1) + torch.sum(exp_logits_2, dim=-1)
    w1 = torch.sum(exp_logits_1, dim=-1) / s
    w2 = torch.sum(exp_logits_2, dim=-1) / s  # bsz x num_heads, tgt_len

    return w1.unsqueeze(-1), w2.unsqueeze(-1)



def adapter_func(x, down_w, up_w, layernorm, training, dropout=0.0, add_residual=True, layernorm_option="in"):
    residual = x

    if layernorm is not None and layernorm_option == "in":
        x = layernorm(x)
    # print("x", x.size())
    # print(x)
    # input()
    # print("down w", down_w.size())
    # print(down_w)
    # input()
    down = x @ down_w
    # print("down", down.size())
    # print(down)
    # input()
    down = nn.functional.relu(down)
    down = nn.functional.dropout(down, p=dropout, training=training)
    up = down @ up_w

    if layernorm is not None and layernorm_option == "out":
        up = layernorm(x)
    if add_residual:
        output = up + residual
    else:
        output = up
    return output

# copied from LoRA: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            lora_init: str="lora",
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.lora_init = lora_init
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.ef_lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.ef_lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'ef_lora_A'):
            if self.lora_init == "bert":
                nn.init.normal_(self.ef_lora_A, std=0.02)
                nn.init.normal_(self.ef_lora_B, std=0.02)
            elif self.lora_init == "lora":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.ef_lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.ef_lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.ef_lora_B @ self.ef_lora_A) * self.scaling
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.ef_lora_B @ self.ef_lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.ef_lora_A.T @ self.ef_lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


# copied from Kernel Adapter for finer control of LoRA

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

class LoRA(nn.Module):
    """Restrict the difference between pre-trained V weights and trained V weights to be low-rank."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        self.q_r = config.lora_Q_rank
        self.v_r = config.lora_V_rank
        
        self.q_share_r = config.lora_Q_share_rank
        self.v_share_r = config.lora_V_share_rank
        
        if self.config.lora_head:
            self.output_dim = self.input_dim // config.num_attention_heads
        else:
            self.output_dim = self.input_dim
        
        if "Q" in self.config.attn_option:
            q_r = self.q_r if config.kerfit_variant_type == "" else self.q_r-1
            q_bias = config.kerfit_variant_type == ""
            self.WQ_LoRA = LoRALinear(self.input_dim, self.output_dim, q_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                # self.config.lora_ini, self.config.lora_head, self.config.lora_head_base,
                self.config.lora_init, self.config.lora_head, self.config.lora_head_base[0],
                q_bias)
            
            if self.q_share_r > 0:
                self.WQ_share_LoRA = LoRALinear(self.input_dim, self.input_dim, self.q_share_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_init, lora_head=False, lora_head_base="X", bias=False)
                
        if "K" in self.config.attn_option:
            q_r = self.q_r if config.kerfit_variant_type == "" else self.q_r-1
            q_bias = False
            self.WQ_LoRA = LoRALinear(self.input_dim, self.output_dim, q_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_init, self.config.lora_head, self.config.lora_head_base[0],
                q_bias)
        
        if "V" in self.config.attn_option:
            # base = self.config.lora_head_base
            # if self.config.lora_head_base == "QS":
                # base = "K"
            
            self.WV_LoRA = LoRALinear(self.input_dim, self.output_dim, self.v_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_init, self.config.lora_head, self.config.lora_head_base[1])
                
            if self.v_share_r > 0:
                self.WV_share_LoRA = LoRALinear(self.input_dim, self.input_dim, self.v_share_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_init, lora_head=False, lora_head_base="X", bias=False)      
        
        if "O" in self.config.attn_option:
            # 64 x 768
            self.WO_LoRA = LoRALinear(self.output_dim, self.input_dim, self.v_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_init, self.config.lora_head, self.config.lora_head_base[-1])
            
            if self.v_share_r > 0:
                self.WO_share_LoRA = LoRALinear(self.input_dim, self.input_dim, self.v_share_r, 
                self.config.lora_alpha, self.config.lora_dropout, self.config.lora_tanh,
                self.config.lora_init, lora_head=False, lora_head_base="A", bias=False)

        
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
                
        
        if "V" in self.config.attn_option and query is not None:
            base = x[self.bases[self.config.lora_head_base[1]]]
            value = value + self.WV_LoRA(base)
            
            if self.v_share_r > 0:
                value = value + self._split_heads(self.WV_share_LoRA(X), value.shape[1], value.shape[3])
            
        if "Q" in self.config.attn_option and query is not None:
            # if self.config.lora_head_base == "QS": X = X - query
            base = x[self.bases[self.config.lora_head_base[0]]]
            query = query + self.WQ_LoRA(base)
            
            if self.q_share_r > 0:
                query = query + self._split_heads(self.WQ_share_LoRA(X), query.shape[1], query.shape[3])
            
        if "K" in self.config.attn_option and key is not None:
            # if self.config.lora_head_base == "QS": X = X - query
            base = x[self.bases[self.config.lora_head_base[0]]]
            query = key + self.WQ_LoRA(base)
        
        if "O" in self.config.attn_option and query is None:
            
            base = x[self.bases[self.config.lora_head_base[-1]]]
            value = value + self.WO_LoRA(base)
            
            if self.v_share_r > 0:
                value = value + self.WO_share_LoRA(X)
        
        return (query, value)

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
        
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

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

class PrefixAdapter(nn.Module):
    """This is the adapter for prefix-tuning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        self.mid_dim = config.prefix_dim
        self.output_dim = self.input_dim // config.num_attention_heads
        
        if (config.prefix_tuning == "landmark" or config.prefix_tuning == "MH_adapter" 
            or config.prefix_tuning == "value_gating"):
            # for value
            self.prefix_v_mlp = PrefixMLP(self.output_dim, config.num_attention_heads, self.mid_dim, 
                                    prefix_value_extension=config.prefix_value_extension,
                                    prefix_head_share=config.prefix_head_share)
            
            if config.prefix_tuning == "value_gating": 
                self.prefix_v_gating_mlp = PrefixMLP(self.output_dim, config.num_attention_heads, self.mid_dim, 
                                    prefix_head_base = 'K',
                                    prefix_value_extension=config.prefix_value_extension,
                                    prefix_head_share=config.prefix_head_share)
            # for key
            elif config.prefix_tuning == "landmark" and config.prefix_length >= 2:
                self.prefix_v_pert_mlp = PrefixMLP(self.output_dim, config.num_attention_heads, config.prefix_length, 
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
