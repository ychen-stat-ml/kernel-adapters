# Implemented based on https://github.com/huggingface/transfer-learning-conv-ai

import sys
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
from typing import Optional, List
from tqdm import tqdm
import warnings
import os

import torch
import torch.nn.functional as F

from transformers import (AdamW, 
                          HfArgumentParser,
                          GPT2Tokenizer, 
                          WEIGHTS_NAME, 
                          CONFIG_NAME, 
                          GPT2Config,
                          set_seed)

from transformers.utils.dummy_pt_objects import NoRepeatNGramLogitsProcessor
from modeling.custom_gpt2 import GPT2LMHeadModel
from training_args import AdapterTrainingArguments

from train import (SPECIAL_TOKENS, TASK_LIST, build_input_from_segments_dialqa, 
    build_input_from_segments_webnlg, build_input_from_segments_mtsm, 
    add_special_tokens_, MAXLEN_MAP, DATASET_PATHS)
from utils import get_dataset_personalities, get_dataset, get_adapter_config
import numpy as np
from dataclasses import dataclass, field

from util.eval_metrics import moses_multi_bleu, rouge#, ent_score

import json

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(tokenizer, model, args, personality=None, history=None, source=None, target=None, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[args.task])
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        if args.task=="qa":
            instance = build_input_from_segments_dialqa(personality, history, current_output, tokenizer, with_eos=False, task=args.task)
        elif args.task == "webnlg":
            instance = build_input_from_segments_webnlg(source, current_output, tokenizer, with_eos=False, task=args.task)
        elif args.task=="summarization":
            # instance, _  = build_input_from_segments_mtsm(source, current_output, tokenizer, with_eos=False, task=args.task)
            instance = build_input_from_segments_mtsm(source, current_output, tokenizer, with_eos=False, task=args.task)
        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        model_outputs = model(input_ids, token_type_ids=token_type_ids, task=args.task)
        logits = model_outputs.get("logits")



        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


@dataclass
class TestingArguments():
    
    seed: int = field(
        default = 123,
        metadata={
            "help": "set seed for training model"
        },
    )
    task: str = field(
        default = "webnlg",
        metadata={"help": "one of task from [qa, summarization, webnlg]"}
    )
    dataset_path: str = field(
        default = None,
        metadata={"help": "Path or url of the dataset."}
    )
    dataset_cache: Optional[str] = field(
        default="./dataset_cache", metadata={"help":"Path or url of the dataset cache"}
    )
    model_checkpoint: Optional[str] = field(
        default="gpt2", metadata={"help": "Path, url or short name of the model"}
    )
    max_history: Optional[int] = field(
        default = 2, metadata={"help":"Number of previous exchanges to keep in history"}
    )

    max_length: Optional[int] = field(
        default = 40, metadata={"help":"Maximum length of the output utterances"}
    )

    min_length: Optional[int] = field(
        default = 0, metadata={"help":"Maximum length of the output utterances"}
    )


    temperature: Optional[float] = field(
        default = 0.7, metadata={"help":"Sampling softmax temperature"}
    )

    top_k: Optional[float] = field(
        default = 0, metadata={"help":"Filter top-k tokens before sampling (<=0: no filtering)"}
    )

    top_p: Optional[float] = field(
        default = 0.9, metadata={"help":"Nucleus filtering (top-p) before sampling (<=0.0: no filtering)"}
    )


    # Flags
    no_sample: bool = field(
        default=True,
        metadata={
            "help": "Do not use token sampling during generation"
        },
    )
    
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={
            "help": "Device (cuda or cpu)"
        },
    )


    def __post_init__(self):
        # Check if the fields are correctly set
        assert self.task in TASK_LIST, "set task amongst TASK_LIST"
        assert self.model_checkpoint != "", "Need to set model checkpoint!"
        # assert self.model_checkpoint != "gpt2", "Set trained model checkpoint from runs folder"



def run():
    
    parser = HfArgumentParser((TestingArguments, AdapterTrainingArguments))

    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, adapter_args = parser.parse_args_into_dataclasses()
    
    # overwrite the original
    # if len(sys.argv) > 2 and sys.argv[2].startswith("runs/"):
    if len(sys.argv) > 2 and "runs/" in sys.argv[2]:
        args.model_checkpoint = sys.argv[2]
        args.seed = int(sys.argv[3])

    # Set the seed for this script
    set_seed(args.seed)

    # Set the dataset path as per the task
    args.dataset_path = DATASET_PATHS[args.task]


    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    

    logger.info("Get pretrained model and tokenizer")
    
    config = GPT2Config.from_pretrained(
             args.model_checkpoint,
             cache_dir="./transformers_cache/",)
    adapter_args.input_dim = config.n_embd
    adapter_args.n_head = config.n_head
    
    adapter_config = get_adapter_config(adapter_args, args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint,
                                              cache_dir="./transformers_cache/",)


    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint, 
                                            config=config,
                                            cache_dir="./transformers_cache/",
                                            adapter_config=adapter_config)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)


    save_path = f"results/{args.task}/{ sys.argv[-1].split('.json')[0].split('/')[-1] }/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # qa interact
    if args.task=="qa":
        output_text = []
        ref_text = []
        loaded_dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.task)
        ## load dev_set_with_ids here
        for pair in tqdm(loaded_dataset["test"]):
            evidence = pair["document"].copy()
            evidence = [evidence[0][:MAXLEN_MAP[args.task]['document']]]
            for utterance in pair["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                with torch.no_grad():
                    out_ids = sample_sequence( tokenizer, model, args, personality=evidence, history=history)
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                output_text.append({"id": pair['id'],"turn_id": utterance['turn_id'],"answer": out_text})
                 
        data = json.dumps(output_text)
        output_file = save_path+args.task+str(args.seed)+"_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(data)
        
        evaluation_command = f"python3 data/CoQA/coqa-evaluate-v1.1.py --data-file data/CoQA/coqa-dev-v1.0.json --pred-file {output_file}"
        
        os.system(evaluation_command)

    if args.task=="webnlg":

        output_text = []
        ref_text = []
        loaded_dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.task)

        prev = None # to identify the refs in the same group
        for pair in tqdm(loaded_dataset["test"]):
            source = pair["src"]
            target = pair["tgt"]
            if not source == prev:
                with torch.no_grad():
                    out_ids = sample_sequence(tokenizer, model, args, source=source, target=target)
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)          
                output_text.append(out_text.replace('\n', ''))
                ref_text.append("")
            
            ref_text.append(tokenizer.decode(pair["tgt"], skip_special_tokens=True))
            prev = source


        sys_file = save_path + str(args.seed) + "_output.txt"
        with open(sys_file, 'w', encoding='utf-8') as f:
            for line in output_text:
                f.write(line)
                f.write('\n')
        
        ref_text = ref_text[1:]
        ref_file = save_path + str(args.seed) + "_ref.txt"
        with open(ref_file, 'w', encoding='utf-8') as f:
            for line in ref_text:
                f.write(line)
                f.write('\n')

        # sys_file = "../../../" + save_path + "_output.txt"
        sys_file = "../../../" + sys_file
        evaluation_command = f"./run_eval_on_webnlg.sh {sys_file} {1}"
        os.chdir("./data/WebNLG/evaluation/")
        os.system(evaluation_command)

        


    if args.task=="summarization":
        output_text = []
        ref_text = []
        loaded_dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.task)
        for pair in tqdm(loaded_dataset["test"]):
            source = pair["src"][:MAXLEN_MAP[args.task]['src']]
            target = pair["tgt"]#[:MAXLEN_MAP[args.task]['tgt']]
            with torch.no_grad():
                out_ids = sample_sequence( tokenizer, model, args, source=source, target=target )
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

            output_text.append(out_text)
            ref_text.append(tokenizer.decode(pair["tgt"], skip_special_tokens=True))
        
        BLEU = moses_multi_bleu(np.array(output_text),np.array(ref_text))
        r_1, r_2, r_l, r_m = rouge(output_text, ref_text)
        print("BLEU:{}".format(BLEU))
        print("ROUGE_1:{}, ROUGE_2:{}, ROUGE_L:{}, ROUGE_mean:{}".format(r_1, r_2, r_l, r_m))
        print("{}\t{}\t{}\t{}\t{}".format(BLEU, r_1, r_2, r_l, r_m))

        with open(save_path+args.task+str(args.seed)+"_output.txt", 'w', encoding='utf-8') as f:
            for line in output_text:
                f.write(line)
                f.write('\n')
        with open(save_path+args.task+str(args.seed)+"_ref.txt", 'w', encoding='utf-8') as f:
            for line in ref_text:
                f.write(line)
                f.write('\n')
        

if __name__ == "__main__":
    run()
    