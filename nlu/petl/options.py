from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenerationArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    min_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "minimal generation length"
        },
    )

    max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "max generation length"
        },
    )

    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "minimal generation length"
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "minimal generation length"
        },
    )

    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "length penalty"
        },
    )

@dataclass
class TuneArguments:
    attn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["prefix", "prefix_nomlp",
            "none", "bitfit", "lora", "adapter", 
            "prompt_tuning", "kernel", "inducer"], \

            "help": "config for attention, none to disable; \
                prefix: mlp reparameterization to output prefix P; \
                prefix_nomlp: prefix P as learned params; \
                adapter: adapter mode; \
                bitfit: the bitfit baseline; \
                lora: the lora baseline; \
                prompt_tuning: the prompt tuning baseline; \
                kernel: kernel-mix; \
                inducer: inducer", 
        },
    )


    attn_option: Optional[str] = field(
        default="concat",
        metadata={
            "choices": ["none", 
                        "concat", 
                        "cross_attn",
                        "cross_attn_noln",
                        "cross_attn_relu",
                        "parallel",
                        "sequential",
                        "QVO",
                        "Qlora",
                        ], \

            "help": "specific attn configs; \
                concat: concat prefix to self, this is prefix tuning baseline; \
                cross_attn_noln: prefix tuning with vanilla add composition (instead of gated add), \
                    need to be used together with 'attn_composition=add'; \
                cross_attn: cross_attn_noln plus a layernorm layer \
                cross_attn_relu: basically multi-head adapter, need to be used under 'prefix' mode; \
                parallel: parallel insertion form; need to be used under 'adapter' mode; \
                sequential: sequential insertion form; need to be used under 'adapter' mode; \
                qvo: train W_q, W_v, and W_o for kernel-mix; \
                Qlora: apply lora to Q before inducer tuning;",

        },
    )

    attn_composition: Optional[str] = field(
        default="add",
        metadata={
            "choices": ["add", "gate_add"],
            "help": "the composition function \
                add: vanilla adding; \
                gate_add: gated adding like prefix tuning"
        },
    )

    ffn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["adapter", "none", "lora"],

            "help": "config for ffn, none to disable; \
            adapter: adapter mode; \
            lora: the lora baseline",
        },
    )

    ffn_option: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["parallel", "sequential", "pfeiffer", "none"], \

            "help": "specific ffn configs; \
                parallel: parallel insertion form; \
                sequential: sequential insertion form; \
                pfeiffer: the Pfeiffer adapter config"
        },
    )


    ffn_adapter_layernorm_option: Optional[str] = field(
        default="in",
        metadata={
            "choices": ["in", "out", "none"],
            "help": "ffn adapter layernorm options; \
                none: no layernorm; \
                in: layernorm applied to input; \
                out: layernorm applied to output"
        },
    )

    ffn_adapter_init_option: Optional[str] = field(
        default="bert",
        metadata={
            "choices": ["bert", "lora"],
            "help": "ffn adapter option"
        },
    )

    ffn_adapter_scalar: Optional[str] = field(
        default="1",
        metadata={
            "help": "the scaling hyperparam for scaled adding composition; \
                set to 'learnable_scalar' to learn this as a parameter"
        },
    )


    mid_dim: Optional[int] = field(
        default=800,
        metadata={
            "help": ""
        },
    )

    attn_bn: Optional[int] = field(
        default=200,
        metadata={
            "help": "the attention bottleneck dimension"
        },
    )

    ffn_bn: Optional[int] = field(
        default=-1,
        metadata={
            "help": "the ffn bottleneck dimension"
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": ""
        },
    )

    unfreeze_params: Optional[str] = field(
        default="ef_",
        metadata={
            "help": "param names that contain the string will \
                be unfreezed, all other params will be freezed"
        },
    )


    load_path: Optional[str] = field(
        default="",
        metadata={
            "help": ""
        },
    )

    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_init: Optional[str] = field(
        default="Houlsby",
        metadata={
            "choices": ["bert", "lora", "Houlsby"],
            "help": "'Houlsby', ini in Houlsby."
        },
    )

    lora_Q_rank: Optional[int] = field(default=4, metadata={"help":"sets the rank for LoRA Q."})
    lora_V_rank: Optional[int] = field(default=4, metadata={"help":"sets the rank for LoRA V."})
    lora_alpha: Optional[float] = field(default=1.0, metadata={"help": "Specify the scaling hyperparameter in LoRA."})
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "Specify the dropout rate in LoRA."})
    lora_tanh: Optional[bool] = field(default=False, metadata={"help": "If set, then update to W becomes tanh(BA)."})
    lora_head: Optional[bool] = field(default=False, metadata={"help": "If set, then update to W will be specific to each head, and its rank = n_head * r."})
    lora_head_base: Optional[str] = field(default="XX", metadata={"help": 'Specify the base for LoRA Q, V, O, "X"/"K"/"V"/"A".'})
    
    lora_Q_share_rank: Optional[int] = field(default=0, metadata={"help":"sets the shared rank for LoRA Q."})
    lora_V_share_rank: Optional[int] = field(default=0, metadata={"help":"sets the shared rank for LoRA V."})

    kerfit_variant_type: Optional[str] = field(default="", metadata={"help": "Specify the type, kernel / bitfit / kerfit."})

    # TODO
    prefix_tuning: Optional[str] = field(default="none", metadata={"help": "Specify the inducer-tuning type, landmark."})
    # prefix_value_extension: Optional[bool] = field(default=False, metadata={"help": "If set, the prefix for V would be extended to W_v and W_o"})

    prefix_dim: Optional[int] = field(
        default=512,
        metadata={
            "help": "the dimension for value prefix"
        },
    )

    prefix_length: Optional[int] = field(
        default=8,
        metadata={
            "help": "the dimension for key prefix"
        },
    )

    prefix_value_extension: Optional[bool] = field(default=False, metadata={"help": "If set, we extend the value to the Wo part."})
    prefix_head_share: Optional[bool] = field(default=False, metadata={"help": "If set, we share the basis among Wo part."})



@dataclass
class MBARTArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dropout: Optional[float] = field(
        default=0.3,
        metadata={
            "help": ""
        },
    )

    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": ""
        },
    )
