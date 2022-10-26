# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch
import torch.nn as nn
from transformers import cached_path
from adapters.adapter_controller import AdapterController
from adapters.adapter_configuration import AutoAdapterConfig
from adapters.adapter_modeling import Adapter, HyperComplexAdapter

from dataclasses import fields


logger = logging.getLogger(__file__)

def get_dataset(tokenizer, dataset_path, dataset_cache=None, task=None, return_cachepath = False):
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__ + '_' + task  # Do avoid using GPT cache for GPT-2 and vice-versa

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)

    else:
        logger.info("Download dataset from %s", dataset_path)
        dataset_file = cached_path(dataset_path)

        if task.lower() == "webnlg":
            # A different loading style for Webnlg corpus. Adopted from PrefixTuning repository (https://github.com/XiangLi1999/PrefixTuning)
            train_dataset = load_webnlg(tokenizer, dataset_path+"train.json")
            valid_dataset = load_webnlg(tokenizer, dataset_path+"dev.json")
            test_dataset = load_webnlg(tokenizer, dataset_path+"test.json")

            dataset = {}
            dataset["train"] = train_dataset
            dataset["valid"] = valid_dataset
            dataset["test"] = test_dataset

        else:
            # Loading scripts for all other datasets from a single json file
            with open(dataset_file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, o) if (n=='id' or n=='turn_id') else (n, tokenize(o)) for n, o in obj.items())
            if isinstance(obj, int):
                return obj
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        
        if dataset_cache:
            torch.save(dataset, dataset_cache)

    if return_cachepath:
        return dataset, dataset_cache
    else:
        return dataset



def load_webnlg(tokenizer, dataset_path):
    '''
    Loads a webnlg 2017 json file and extracts the triples and target sentence.
    '''

    logger.info("Creating features from dataset file at %s", dataset_path)

    with open(dataset_path) as f:
        lines_dict = json.load(f)

    dataset = []

    for i, example in enumerate(lines_dict['entries']):
        sents = example[str(i + 1)]['lexicalisations']
        triples = example[str(i + 1)]['modifiedtripleset']

        temp_triples = []
        for j, tripleset in enumerate(triples):
            subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
            triples = {}
            triples["subject"] = subj
            triples["property"] = rela
            triples["object"] = obj
            temp_triples.append(triples)

        for sent in sents:
            if sent["comment"] == 'good':
                data = {}
                data["src"] = temp_triples
                data["tgt"] = sent["lex"]
                dataset.append(data)

    return dataset


    # edited_sents = []
    # for src, tgt in zip(full_src_lst, full_tgt_lst):
    #     sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
    #     edited_sents.append(sent)

    # batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
    #                             is_split_into_words=False)
    # self.examples = batch_encoding["input_ids"]






    # assert len(self.src_cat) == len(self.examples)

    # pass

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None, task=None):
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__ + '_' + task # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)

    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        dataset_file = cached_path(dataset_path)
        with open(dataset_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                if ('id' in obj.keys()) or ('turn_id' in obj.keys()):
                    return obj
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)
    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str, task, lr, n_epochs):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name + '_' + task + '_' + str(lr) + '_epoch' + str(n_epochs))
    return logdir


def get_adapter_config(adapter_args, training_args):
    if adapter_args.train_task_adapters or adapter_args.kernel_adapter or adapter_args.lora != "" or adapter_args.prefix_tuning:
        
        adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
        # adapter_config.input_dim = 768 # TODO: use model.config.d_model to set this
        adapter_config.input_dim = adapter_args.input_dim
        adapter_config.n_head = adapter_args.n_head
        

        adapter_config.tasks = [training_args.task]

        adapter_params = [field.name for field in fields(adapter_args)]
        for p in adapter_params:
            # print(p, hasattr(adapter_args, p), hasattr(adapter_config, p))
            if hasattr(adapter_args, p) and hasattr(adapter_config, p) and\
                    getattr(adapter_args, p) is not None:
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = training_args.device
    else:
        adapter_config = None

    return adapter_config



def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_model_params(model, adapter_args):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      adapter_args: defines the adapters arguments.
    """
    # If we are training adapters, we freeze all parameters except the
    # adapter parameters like adapter controllers.
    
    if not adapter_args.unfreeze_all:
        freeze_params(model)
    
    if adapter_args.train_task_adapters:
        # freeze_params(model)

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController, Adapter)):
                if isinstance(sub_module, (AdapterController, HyperComplexAdapter)) and adapter_args.hypercomplex_adapters:
                    for param_name, param in sub_module.named_parameters():
                        if any(x in param_name for x in ["phm_rule", "phm_rule_left", "phm_rule_right"]) and not adapter_args.learn_phm:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                else:
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        
        if adapter_args.hypercomplex_adapters and adapter_args.shared_phm_rule:
            if adapter_args.factorized_phm_rule:
               model.phm_rule_left.requires_grad = True
               model.phm_rule_right.requires_grad = True
            else:
               model.phm_rule.requires_grad = True
                 
        if adapter_args.hypercomplex_adapters and adapter_args.shared_W_phm:
            if adapter_args.factorized_phm:
               model.W_down_left.requires_grad = True
               model.W_down_right.requires_grad = True
               model.W_up_left.requires_grad = True
               model.W_up_right.requires_grad = True
            else:
               model.W_down.requires_grad = True
               model.W_up.requires_grad = True

    # Proposed kernel adapter fixes everything except the attention bandwith parameters.
    if adapter_args.kernel_adapter:
        # freeze_params(model)

        # unfreeze bandwidth terms.
        for n,p in model.named_parameters():
          # if "bandwidth" in n:
          if "adapter_controller_k" in n:
            p.requires_grad = True

    # Unfreeze prefix tuning parameters
    if adapter_args.prefix_tuning != "":
        for n, m in model.named_parameters():
            # if "prefix_shared" == n:
            # print(n)
            if "prefix" in n:
                m.requires_grad = True
            
    # For bitfit we freeze the whole model except for the biases and the final classifier layer.
    if adapter_args.bitfit: 
        # freeze_params(model)

        # unfreeze bias terms.
        for n,p in model.named_parameters():
          if ".bias" in n:
            p.requires_grad = True

        # unfreeze the classifier.
        # TODO: Need to check if bitfit (or compacter paper) unfreezes LM head by default 
        for param in model.lm_head.parameters():
            param.requires_grad = True

        if adapter_args.freeze_bitfit_lm_head:
           for n, param in model.lm_head.named_parameters():
                if "bias" in n:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
        if adapter_args.freeze_bitfit_lm_head_all:
           for n, param in model.lm_head.named_parameters():
                param.requires_grad = False
    
    if adapter_args.lora != "":
        for n,p in model.named_parameters():
          if "adapter_controller_l" in n:
            p.requires_grad = True
    
    # Unfreezes layer norms.
    if adapter_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (nn.LayerNorm)):
                if len(name.split(".")) < 6: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    # Unfreezes last linear layer of decoder (which is tied to the input vocab embeeding).
    if adapter_args.unfreeze_lm_head or adapter_args.unfreeze_task_embeddings:
        for param in model.lm_head.parameters():
              param.requires_grad = True




def modify_model_after_init(model, training_args, adapter_args):
    # Freezes model parameters.
    freeze_model_params(model, adapter_args)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Model Trainable Parameters {} *****".format(trainable_params))
    print("***** Model Trainable Parameters {} *****".format(trainable_params))
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("inside n ", n, p.numel())
    
    
    
    
    
    # if training_args.print_num_parameters:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             logger.info("##### Parameter name %s", name)
    #     total_lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    #     total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     total_trainable_bias_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.endswith(".b"))
    #     total_trainable_layernorm_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ".layer_norm.weight" in n)
    #     total_params = sum(p.numel() for p in model.parameters())
    #     logger.info("Total trainable parameters %s", total_trainable_params)
    #     logger.info("Total traianable bias parameters %s", total_trainable_bias_params)
    #     logger.info("Total trainable layernorm parameters %s", total_trainable_layernorm_params)
    #     logger.info("Total parameters %s", total_params)
    #     t5_base_params=222882048
    #     # total params since we have 8 task, it is Y = 1*BERT + 8*ADAPTERS, and final number is Y/BERT ("1.3x")
    #     total_params_ratio = ((total_params-t5_base_params)*8+t5_base_params)/t5_base_params
    #     total_trainable_params_percent =(total_trainable_params/t5_base_params)*100
    #     total_trainable_bias_params_percent =(total_trainable_bias_params/total_trainable_params)*100
    #     total_trainable_layernorm_params_percent =(total_trainable_layernorm_params/total_trainable_params)*100
    #     total_trainable_lm_head_params_percent =(total_lm_head_params/t5_base_params)*100
    #     logger.info("For adapters/prompt-tuning, total params %s", total_params_ratio)
    #     logger.info("For intrinsic, total params %s", total_params/t5_base_params)
    #     logger.info("Total trainable params %s", total_trainable_params_percent)
    #     logger.info("Total trainable bias params %s", total_trainable_bias_params_percent)
    #     logger.info("Total trainable layernorm params %s", total_trainable_layernorm_params_percent)
    #     logger.info("Total lm_head params %s", total_trainable_lm_head_params_percent)
    return model

