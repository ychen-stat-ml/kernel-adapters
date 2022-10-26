# Implemented based on https://github.com/huggingface/transfer-learning-conv-ai
##############################################################################
# set the gpu to be used
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

##############################################################################

import sys
import math
import json
import logging
from pprint import pformat
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from itertools import chain
from typing import Optional, List
import numpy as np



import torch


import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

from transformers import (AdamW, 
                          HfArgumentParser,
                          GPT2Tokenizer, 
                          Seq2SeqTrainingArguments,
                          WEIGHTS_NAME, 
                          CONFIG_NAME, 
                          GPT2Config,
                          set_seed)

from modeling.custom_gpt2 import GPT2LMHeadModel
from training_args import AdapterTrainingArguments
from adapters import ADAPTER_CONFIG_MAPPING
from dataclasses import dataclass, field

from utils import get_dataset, make_logdir, get_adapter_config, modify_model_after_init


##############################################################################
# Tokens and other settings

SPECIAL_TOKENS = {"qa":["<bos_qa>", "<eos_qa>", "<question>", "<answer>", "<document>"], 
                  "summarization":["<bos_sm>", "<eos_sm>", "<source_sm>", "<target_sm>"],
                  "webnlg":["<bos_webnlg>", "<eos_webnlg>", "<subject>", "<property>", "<object>", "<target_webnlg>"]
                  }

ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>', 'additional_special_tokens': (
                  "<bos_qa>", "<eos_qa>", "<question>", "<answer>", "<document>", 
                  "<bos_sm>", "<eos_sm>", "<source_sm>", "<target_sm>",
                  "<bos_webnlg>", "<eos_webnlg>", "<subject>", "<property>", "<object>", "<target_webnlg>",
                  "<pad>",)}

TASK_LIST = ["qa", "summarization", "webnlg"]

MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

MAXLEN_MAP = {"summarization":{'src':400, 'tgt':130}, 
    "qa":{'document':400, 'qa':50}, "webnlg":{'src':400, 'tgt':130}}


DATASET_PATHS = {"summarization":"data/CNNDAILY/data.json",
                "qa":"data/CoQA/data2.json", 
                "webnlg":"data/WebNLG/"}

#######################################################################################



logger = logging.getLogger(__file__)

##############################################################################


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # returns 0 and doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments_dialqa(persona, history, reply, tokenizer, lm_labels=False, with_eos=True, task=None):
    """ Build a sequence of input from 3 segments for dialogue and coqa task: persona(document), history(question) and last reply(answer). """
    bos, eos, speaker1, speaker2, persona_token= tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
    instance = {}
    sequence = [[bos] + [persona_token] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [persona_token for _ in sequence[0]] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    # print("input:", tokenizer.decode(instance["input_ids"]))
    # print("type:", tokenizer.decode(instance["token_type_ids"]))
    # print("lm_label:", instance["lm_labels"])
    # print("==================================================")
    return instance

def build_input_from_segments_webnlg(source, target, tokenizer, lm_labels=False, with_eos=True, task=None):
    """ Build a sequence of input from 2 segments: source and target for natrual langugage generation task og Webnlg 2017. """
    bos, eos, subject, property, object, tgt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
    nlg_map = {"subject":subject, "property":property, "object":object}
    instance = {}

    source_list = []
    for source_sample in source:
        for key in ["subject", "property", "object"]:
            source_list.append([nlg_map[key]] + source_sample[key])

    sequence = [[bos]] + source_list + [[tgt] + target + ([eos] if with_eos else [])]
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [bos] + list(chain(*[[s[0]]*len(s) for s in sequence[1:-1]])) + [tgt]* len(sequence[-1])
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    assert len(instance["input_ids"]) == len(instance["token_type_ids"])
    assert len(instance["input_ids"]) == len(instance["lm_labels"])
    # print("input:", tokenizer.decode(instance["input_ids"]))
    # print("type:", tokenizer.decode(instance["token_type_ids"]))
    # print("lm_label:", instance["lm_labels"])
    # print("==================================================")
    return instance

def build_input_from_segments_mtsm(source, target, tokenizer, lm_labels=False, with_eos=True, task=None):
    """ Build a sequence of input from 2 segments: source and target for machine translation and summarization task. """
    bos, eos, src, tgt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
    instance = {}
    sequence = [[bos]] + [[src] + source] + [[tgt] + target + ([eos] if with_eos else [])]
    instance["input_ids"] = list(chain(*sequence))

    if len(instance["input_ids"])==0:
        print("yes")

    instance["token_type_ids"] = [bos] + [src]* len(sequence[1]) + [tgt]* len(sequence[2])
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    return instance


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    loaded_dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.task)
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}

    def load_qa():
        for dataset_name, dataset in loaded_dataset.items():
            tgt = "candidates"
            for pair in dataset:
                evidence = pair["document"].copy()
                evidence = [evidence[0][:MAXLEN_MAP[args.task]['document']]]
                for utterance in pair["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    candidate = utterance[tgt][-1] #only last one candidate which is gold response
                    lm_labels = True # always train language model
                    instance = build_input_from_segments_dialqa(evidence, history, candidate, tokenizer, lm_labels, task=args.task)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)

    def load_webnlg():
        for dataset_name, dataset in loaded_dataset.items():
            for pair in dataset:
                source = pair["src"]
                target = pair["tgt"]
                instance = build_input_from_segments_webnlg(source, target, tokenizer, lm_labels=True, with_eos=True, task=args.task)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)


    def load_mtsm():
        for dataset_name, dataset in loaded_dataset.items():
            if dataset_name=="val":
                dataset_name="valid" # fix dataset key error
            tgt = "tgt"
            for pair in dataset:
                if len(pair["src"])<2 or len(pair[tgt])<2: # filter the shot sentence
                    continue
                source = pair["src"][:MAXLEN_MAP[args.task]['src']]
                target = pair[tgt][:MAXLEN_MAP[args.task]['tgt']]
                instance =  build_input_from_segments_mtsm(source, target, tokenizer, lm_labels=True, with_eos=True, task=args.task)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
    logger.info("Pad inputs and convert to Tensor")
    

    for dataset_name, dataset in loaded_dataset.items():
        print(dataset_name, len(dataset))
    
    if args.task =="qa":
        load_qa()
        _ = datasets.pop("test", None)
        tensor_datasets = {"train": [], "valid": []}
    elif args.task =="webnlg":
        load_webnlg()
        tensor_datasets = {"train": [], "valid": [], "test": []}
    else:
        load_mtsm()
        tensor_datasets = {"train": [], "valid": [], "test": []}

    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN['pad_token']))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])

            tensor = tensor.view((-1, 1) + tensor.shape[1:]) #always one candidate
            tensor_datasets[dataset_name].append(tensor)



    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler
    
def count_parameters(list_param):
    return sum(p.numel() for p in list_param)

def count_parameters_model(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train():
    
    # Training arguments loaded using normal ArgumentParser
    # TODO: eventually move this to HFArgumentParser. But currently, local_rank is not correctly set when run in distriuted mode. 
    # Hence using ArgumentParser presently.
    parser = ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, default="configs/baseline.json", help="Set the config for the train. Default config is baseline")


    parser.add_argument("--seed", type=int, help="set seed for training model")
    parser.add_argument("--task", type=str, default="summarization", help="one of task from [qa, summarization, webnlg]")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    

    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--label_smooth", type=float, default=0.0, help="label smoothing weight")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--early_stop_metric", type=str, default="average_ppl", help="How to decide early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for early stopping.")



    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    parser.add_argument("--random_init", action='store_true', help="don't use GPT-2 pre-trained model ")
    parser.add_argument("--distillation", action='store_true')

    args = parser.parse_args()

    # Update the training and adapter args as per config
    with open(os.path.abspath(args.config), "r") as f:
            json_params = json.load(f)
            args = vars(args)
            for key, val in json_params.items():
                if key in args:
                    args[key] = val
            args = Namespace(**args)


    # Adapter arguments loaded using Dataclasses and HFArgumentParser
    adapter_parser = HfArgumentParser(AdapterTrainingArguments)
    adapter_args, = adapter_parser.parse_json_file(args.config)


    # Sanity checks
    assert args.task is not None, "Task is not set."
    assert args.task in TASK_LIST, "set task amongst TASK_LIST"
    assert args.seed is not None
    
    
    # Set seed before initializing model.
    set_seed(args.seed) # set seed for random / numpy / pytorch / tensorflow
    next_seed = np.random.randint(1120)
    
    # Set the dataset path as per the task
    args.dataset_path = DATASET_PATHS[args.task]                                                


    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))
    
    
    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    
    
    # specify the model to run
    # model_class = GPT2LMHeadModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    
    config = GPT2Config.from_pretrained(
             args.model_name,
             cache_dir="./transformers_cache/",)
    adapter_args.input_dim = config.n_embd
    adapter_args.n_head = config.n_head
    
    config.train_task_adapters = adapter_args.train_task_adapters
    config.kernel_adapter = adapter_args.kernel_adapter
    config.prefix_tuning = adapter_args.prefix_tuning
    
    
    adapter_config = get_adapter_config(adapter_args, args)
    # print(adapter_config.train_task_adapters)


    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name,
                                              cache_dir="./transformers_cache/",)


    model = GPT2LMHeadModel.from_pretrained(args.model_name, 
                                            config=config,
                                            cache_dir="./transformers_cache/",
                                            adapter_config=adapter_config)
    
    
    model.to(args.device)

    # Add special tokens if they are not already added
    adapter_args.vocab_size_original= model.config.vocab_size
    add_special_tokens_(model, tokenizer)

    ##########################################################################
    # Manages which parameters to fix in the model

    # Modify model trainable parameters as per adapter settings
    model = modify_model_after_init(model, args, adapter_args)

    # Keep a list of the original frozen parameters (To check if they get updated)
    # param_to_check0 = []
    # for n, p in model.named_parameters():
        # if p.requires_grad == False:
            # param_to_check0.append(p.clone())
        # if "transformer.wte.weight" in n:
            # param_to_check0.append(p[:adapter_args.vocab_size_original,:].clone())
    # frozen_params = list(param_to_check0)

    # weight_decay > 0 will update the weights even its requires_grad = False
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, correct_bias=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)


    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], 
            # output_device=args.local_rank)
            output_device=args.local_rank, find_unused_parameters=True)


    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)


    ##########################################################################
    # Use ignite.engine to define the modules for training
    
    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, lm_labels, token_type_ids = batch

        # Check that input doesn't have NaNs
        assert not torch.isnan(input_ids).any()
        assert not torch.isnan(lm_labels).any()
        assert not torch.isnan(token_type_ids).any()

        model_outputs = model(
            input_ids, token_type_ids=token_type_ids, 
            # labels=lm_labels, label_smooth=args.label_smooth, task=args.task
            labels=lm_labels, label_smooth=args.label_smooth, task=args.task
        )

        lm_loss = model_outputs.get("loss") # crossEntropy

        loss = lm_loss/ args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        
        if engine.state.iteration % args.gradient_accumulation_steps == 0:

            if adapter_args.unfreeze_task_embeddings:
                # Manually set the gradients of the normal tokens to be zero
                # This ensures that only the task embedding tokens are trained (when optimizer.step() is called in the next line). 
                if args.distributed:
                    model.module.transformer.wte.weight.grad[:adapter_args.vocab_size_original,:] = 0
                else:
                    model.transformer.wte.weight.grad[:adapter_args.vocab_size_original,:] = 0

            optimizer.step()

            # Sanity check that frozen parameters are not updated by optimizer momentum terms
            # param_to_check = []
            # for n, p in model.named_parameters():
                # if p.requires_grad == False:
                    # param_to_check.append(p.clone())
                # if "transformer.wte.weight" in n:
                    # param_to_check.append(p[:adapter_args.vocab_size_original,:].clone())
            # params_after_optimizer = list(param_to_check)

            # for param_old, param_new in zip(frozen_params, params_after_optimizer):
                # assert torch.equal(param_old.data, param_new.data)

            optimizer.zero_grad()

        return loss.item()

    trainer = Engine(update)


    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, lm_labels, token_type_ids = batch

            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            model_outputs = model(
                input_ids, token_type_ids=token_type_ids, task=args.task
            )
            lm_logits = model_outputs.get("logits")

            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, ), (lm_labels_flat_shifted, )

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    
    # add a warmup phase to the lr scheduler
    scheduler = create_lr_scheduler_with_warmup(scheduler,
                                            warmup_start_value=0.0,
                                            warmup_end_value=args.lr,
                                            warmup_duration=500)
                                            # warmup_end_value=6.25e-5,
                                            # warmup_duration=100)
    
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics.update({"average_ppl": MetricsLambda(math.exp, metrics["average_nll"])})
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    ##########################################################################
    # set up early stopping criterion - PPL should reduce AND ACCURACY should increase
    def early_stopping_score_function(engine):
        if args.early_stop_metric == "average_ppl":
            logger.info(engine.state.metrics.keys())
            all_reduced_ppl = engine.state.metrics["average_ppl"]
            es_metric = -all_reduced_ppl
        elif args.early_stop_metric == "average_accuracy":
            all_reduced_acc = engine.state.metrics["average_accuracy"]
            es_metric = all_reduced_acc
        else:
            raise ValueError("Unsupported metric for early stopping!")
        return es_metric


    valid_es_handler = EarlyStopping(patience=args.patience, 
                                     score_function=early_stopping_score_function,
                                     trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, valid_es_handler)
    
    ##########################################################################
    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        # log_dir = make_logdir(args.model_name, task= args.task, lr=args.lr, n_epochs = args.n_epochs)
        model_identifier = args.config.split("/")[-1][:-5]
        log_dir = make_logdir(model_identifier, task= args.task, lr=args.lr, n_epochs = args.n_epochs)
        if args.distillation:
            log_dir+="_distillation"

        tb_logger = TensorboardLogger(log_dir)
        
        import shutil; shutil.copy(args.config, log_dir)
        with open(os.path.abspath(args.config), "w", encoding='utf-8') as f:
            json_params["seed"] = next_seed
            json_params["model_checkpoint"] = log_dir
            json.dump(json_params, f, ensure_ascii=False, indent=4)
            
        tb_logger.attach(trainer, 
                        log_handler=OutputHandler(tag="training", 
                        metric_names=["loss"]), 
                        event_name=Events.ITERATION_COMPLETED)

        tb_logger.attach(trainer, 
                        log_handler=OptimizerParamsHandler(optimizer), 
                        event_name=Events.ITERATION_STARTED)

        tb_logger.attach(evaluator, 
                        log_handler=OutputHandler(tag="validation", 
                        metric_names=list(metrics.keys()), 
                        global_step_transform=global_step_from_engine(trainer)), 
                        event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 
                                             'checkpoint', 
                                             score_function=ModelCheckpoint.get_default_score_fn("average_ppl", -1.0),
                                             n_saved=3)

        # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)}) 

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)
    
    ##########################################################################
    # Run the training
    """
    if args.epoch_length: # for debugging
        trainer.run(train_loader, max_epochs=args.n_epochs, epoch_length=args.epoch_length)
    else:
    """
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir,checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
    # model = train()
