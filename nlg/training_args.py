from adapters import ADAPTER_CONFIG_MAPPING
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AdapterTrainingArguments:
    """Defines the adapters parameters."""
    train_task_adapters: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds task adapters in the model."})
    adapter_config_name: Optional[str] = field(
        default="adapter", metadata={"help": "config name for the adapter layers, should be selected "
        f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."}
    )
    add_layer_norm_before_adapter: Optional[bool] = field(default=False, metadata={
        "help": "whether to have layer-norm before adapter."})
    add_layer_norm_after_adapter: Optional[bool] = field(default=False,
        metadata={"help": "whether to have layer-norm after adapter."})
    hidden_dim: Optional[int] = field(default=128, metadata={"help": "defines the default hidden dimension for "
        "adapter layers."})
    task_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "adapter layers."})
    task_reduction_factor_list: Optional[List[int]] = field(
        default = None, metadata={"help": "defines the specific reduction factor for each adapter layers."})
    non_linearity: Optional[str] = field(default="swish", metadata={"help": "Defines nonlinearity for adapter layers."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "If set unfreeze the last linear layer."})
    unfreeze_task_embeddings: bool = field(default=False, metadata={"help": "If set unfreeze, the special tokens (task embeddings) are also trained."})
    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "If set, unfreezes the layer norms."})
    task_adapter_layers_decoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the decoder."})
    add_adapter_in_feed_forward: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the feedforward."})
    add_adapter_in_self_attention: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the selfattention"})
    
    parallel_adapter: Optional[str] = field(default="", metadata={"help": "If set, use the parallel adapter."})



    # PHM and compacter arguments
    hypercomplex_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the hypercomplex layers"
                                                                                "for adapters."})
    hypercomplex_division: Optional[int] = field(default=8, metadata={"help": "Defines the number to divide the dimensions"})
    
    learn_phm: Optional[bool] = field(default=True, metadata={"help": "If set, learns the phm rules in Hypercomplex adapters."})
    normalize_phm_weight: Optional[bool] = field(default=False, metadata={"help": "Weather to normalize the weights of"})

    hypercomplex_nonlinearity: Optional[str] = field(default="glorot-uniform", metadata={"help": "Defines the nonlinearity for the"
        " hypercomplex adapter layers."})
    shared_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, uses a shared phm rules for all"
        " hypercomplex adapter layers."})
    factorized_phm: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the weights for the W in"
        " hypercomplex adapters."})
    shared_W_phm: Optional[bool] = field(default=False, metadata={"help": "If set, shares the W in phm adapter layers between all adapters."})
    factorized_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the shared weights for the W in"
        " hypercomplex adapters."})
    phm_c_init: Optional[str] = field(default="normal", metadata={"help": "Initialization for the phm rules."})
    phm_rank: Optional[int] = field(default=1, metadata={"help":"sets the rank for the phm decomposition."})
    phm_init_range: Optional[float] = field(default=0.01, metadata={"help": "defines the phm init range."})
    
    kronecker_prod: Optional[bool] = field(default=False, metadata={"help": "If set, compute the kronecker using another version."})


    # Bitfit arguments
    bitfit: Optional[bool] = field(default=False, metadata={"help": "If set, we train the bitfit model."})
    freeze_bitfit_lm_head: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
    freeze_bitfit_lm_head_all: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})



    # Proposed Kernel Adapter arguments
    kernel_adapter: Optional[bool] = field(default=False, metadata={"help": "If set, we train the kernel adapter."})
    shared_bandwidth: Optional[bool] = field(default=True, metadata={"help": "If set, we share the same kernel adapter among different heads."})
    kernel_bitfit_variant: Optional[bool] = field(default=False, metadata={"help": "If set, we adjust the bias of Q and K"})
    corresponding_kernel: Optional[bool] = field(default=False, metadata={"help": "If set, only the layer with adapter will have kernel adapter."})
    kernel_Q_only: Optional[bool] = field(default=False, metadata={"help": "If set, only the adapter for Q will be used."})
    kerfit_variant_type: Optional[str] = field(default="", metadata={"help": "Specify the type, kernel / bitfit / kerfit."})
    
    # LoRA
    lora: Optional[str] = field(default="", metadata={"help": 'Specify the weight matrices to update, ""/"Q"/"V"/"QV".'})
    lora_Q_rank: Optional[int] = field(default=4, metadata={"help":"sets the rank for LoRA Q."})
    lora_V_rank: Optional[int] = field(default=4, metadata={"help":"sets the rank for LoRA V."})
    lora_alpha: Optional[float] = field(default=1.0, metadata={"help": "Specify the scaling hyperparameter in LoRA."})
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "Specify the dropout rate in LoRA."})
    lora_tanh: Optional[bool] = field(default=False, metadata={"help": "If set, then update to W becomes tanh(BA)."})
    lora_ini: Optional[str] = field(default="He_s", metadata={"help": 'Specify the initilaization, "He", original LoRA/"He_s", original LoRA w/ smaller scale, /"Houlsby", ini in Houlsby.'})
    lora_head: Optional[bool] = field(default=False, metadata={"help": "If set, then update to W will be specific to each head, and its rank = n_head * r."})
    lora_head_base: Optional[str] = field(default="XX", metadata={"help": 'Specify the base for LoRA Q, V, O, "X"/"K"/"V"/"A".'})
    
    lora_Q_share_rank: Optional[int] = field(default=0, metadata={"help":"sets the shared rank for LoRA Q."})
    lora_V_share_rank: Optional[int] = field(default=0, metadata={"help":"sets the shared rank for LoRA V."})
    
    # flag for full finetuning
    unfreeze_all: Optional[bool] = field(default=False, metadata={"help": "If set, we unfreeze all the parameters."})
    
    # flag for low-std initialization
    truncated_gaussian_initialization: Optional[bool] = field(default=False, metadata={"help": "If set, we initialize the adapter parameters with trancated N[0, 0.01^2]."})
    initialization_scale: Optional[str] = field(default="0.01", metadata={"help": "Specify the initialization type, ntk / 0.01, neural tangent kernel (1/bottleneck) or a certain number."})

    # Prefix Tuning
    prefix_tuning: Optional[str] = field(default="", metadata={"help": "Specify the prefix tuning mode."})
    prefix_length: Optional[int] = field(default=8, metadata={"help": "Specifies the prefix sequence length, 0 for adaptive length, >1 for the bottleneck size of MLP for prefix_perturbation"})
    prefix_dim: Optional[int] = field(default=512, metadata={"help": "Specifies the prefix embedding dimension."})
    
    prefix_value_extension: Optional[bool] = field(default=False, metadata={"help": "If set, we extend the value to the Wo part."})
    prefix_head_share: Optional[bool] = field(default=False, metadata={"help": "If set, we share the basis among Wo part."})
    prefix_gating_ablation: Optional[bool] = field(default=False, metadata={"help": "If set, compute (1-w) f(q) + w P_{v,i}, instead of f(q) + w P_{v,i}."})
    
    # prefix_perturbation: Optional[bool] = field(default=False, metadata={"help": "If set, we model the prefix as Q + MLP(Q)."})
    
    # TODO
    init_prefix_from_vocab: Optional[bool] = field(default=False, metadata={"help": "Initialize prefix from the tokens of pretrained gpt model."})