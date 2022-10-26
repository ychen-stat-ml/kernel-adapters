export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}


# TASK_NAME=mnli
TASK_NAME=sst2
metric="accuracy"

# wandb env variables
export WANDB_PROJECT=glue.${TASK_NAME}
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`

seed=42

# ----- MAM adapter -----
# attn_mode="prefix"
# attn_option="concat"
# attn_composition="add"
# attn_bn=16  # attn bottleneck dim

# ffn_mode="adapter"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="2"
# ffn_bn=16 # ffn bottleneck dim

# prefix_tuning="none"
# prefix_length=0
# prefix_dim=0
# prefix_value_extension=false

# ----- Kernel-mix -----
# attn_mode="kernel"
# attn_option="QVO"
# attn_composition="add"
# attn_bn=16  # attn bottleneck dim

# ffn_mode="none"
# ffn_option="none"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="2"
# ffn_bn=16 # ffn bottleneck dim

# lora_alpha=-1
# lora_Q_rank=1
# lora_V_rank=2
# lora_Q_share_rank=2
# lora_V_share_rank=4
# lora_dropout=0.1
# lora_init="Houlsby"
# lora_head=true
# lora_head_base="XKA"

# ----- lora -----
# attn_mode="lora"
# attn_option="none"
# attn_composition="add"
# attn_bn=16

# set ffn_mode to be 'lora' to use
# lora at ffn as well

# ffn_mode="lora"
# ffn_option="none"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="bert"
# ffn_adapter_scalar="1"
# ffn_bn=16

# lora_alpha=32
# lora_dropout=0.1
# lora_init="lora"

# ----- inducer + lora -----
attn_mode="inducer"
attn_option="Qlora"
attn_composition="add"
attn_bn=4

ffn_mode="none"
ffn_option="none"
ffn_adapter_layernorm_option="none"
ffn_adapter_scalar="1"
ffn_adapter_init_option="bert"
ffn_bn=0

lora_alpha=32
lora_dropout=0.1
lora_init="lora"

prefix_tuning="landmark"
prefix_length=2
prefix_dim=4
prefix_value_extension=true

# ----- inducer + para_ffn -----
# attn_mode="inducer"
# attn_option="Qlora"
# attn_composition="add"
# attn_bn=4


# ffn_mode="adapter"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="2"
# ffn_bn=12


# lora_alpha=32
# lora_dropout=0.1
# lora_init="lora"

# prefix_tuning="landmark"
# prefix_length=2
# prefix_dim=2
# prefix_value_extension=true

# ----- inducer w/o lora -----
# attn_mode="inducer"
# attn_option="none"
# attn_composition="add"
# attn_bn=4

# ffn_mode="none"
# ffn_option="none"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_scalar="1"
# ffn_adapter_init_option="bert"
# ffn_bn=0

# lora_alpha=32
# lora_dropout=0.1
# lora_init="lora"

# prefix_tuning="landmark"
# prefix_length=6
# prefix_dim=4
# prefix_value_extension=true
#########################################################


# lora params are not set
if [ -z ${lora_alpha+x} ];
then
    lora_alpha=0
    lora_init="lora"
    lora_dropout=0
fi

# set to 1 for debug mode which only
# uses 1600 training examples
debug=1

# set to "wandb" to use weights & bias
report_to="none"

bsz=16
gradient_steps=1

lr=2e-4
max_grad_norm=1
# weight_decay=0
weight_decay=0.1
warmup_updates=0
warmup_ratio=0.06
max_steps=-1
num_train_epochs=10
max_tokens_per_batch=0
max_seq_length=512

lr_scheduler_type="polynomial"
#metric=bleu
unfreeze='ef_'
max_eval_samples=1600
logging_steps=50

eval_strategy="epoch"
save_steps=5000

extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    weight_decay=0
    max_grad_norm=1
    max_train_samples=1000
    max_eval_samples=150
    bsz=16
    gradient_steps=1
    num_train_epochs=2
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples} --max_predict_samples 150"
    debug_str=".debug"
fi



# for seed in "${seed_list[@]}"; do

exp_name=glue.${TASK_NAME}.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name+=.fo_${ffn_option}.abn${preseqlen}.fbn${ffn_bn_len}.ac_${attn_composition}
exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}
exp_name+=.fs_${ffn_adapter_scalar}.unfrz_${unfreeze}.ne${num_train_epochs}
exp_name+=.bsz${bsz}.grad_steps${gradient_steps}.lr${lr}
exp_name+=.warm${warmup_ratio}.wd${weight_decay}.seed${seed}.${debug_str}
SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}00
echo "${SAVE}"
rm -rf ${SAVE}; mkdir -p ${SAVE}

rm checkpoints/hf_model/downloads/*.lock
rm checkpoints/hf_model/*.lock

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name $TASK_NAME \
    --model_type roberta_custom \
    --do_train \
    --do_eval \
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --max_tokens_per_batch ${max_tokens_per_batch} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --prefix_tuning ${prefix_tuning} \
    --prefix_length ${prefix_length} \
    --prefix_dim ${prefix_dim} \
    --prefix_value_extension ${prefix_value_extension} \
    --seed ${seed} \
    --unfreeze_params ${unfreeze} \
    --max_eval_samples ${max_eval_samples} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --warmup_ratio ${warmup_ratio} \
    --max_seq_length ${max_seq_length} \
    --fp16 \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to ${report_to} \
    --run_name ${TASK_NAME}.${DATE}.${exp_name} \
    --overwrite_output_dir \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --ddp_find_unused_parameter "False" \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt
# done
