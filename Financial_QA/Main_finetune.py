import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../../../')
sys.path.append('../../../../../../')
sys.path.append('../common')
# print(sys.path)
import config
from PrivLM_Bench.training_interface import DP_DDP_trainer, DDP_trainer, DDP_QLora_trainer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel, LoraConfig
import json
from torch.nn.utils.rnn import pad_sequence
import argparse
import importlib
from Evaluate.evaluate_metrics import Meter
import deepspeed
import hjson
import random
from torch.utils.data.distributed import DistributedSampler
from deepspeed import comm as dist 
import time
import numpy as np
from Dataset.finqa_dataset import SingleR_Train_Dataset, SingleR_Train_Dataset_Orpo, collate_fn_Train, collate_fn_Train_orpo
TIME_STR = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
import pickle
import math

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

    
class Llama_Finetune(DDP_trainer):
    def __init__(self, config):
        self.meter = Meter()
        self.init_model_dir = config['init_model_dir']

        #* add new parameter here
        super().__init__(**config)
        self.model_save_dir = os.path.join(config['model_save_dir'], f"BEST_MODEL")
        self.best_rl_path = config['best_rl_path']
        if os.path.exists(config['best_rl_path']):
            with open(config['best_rl_path'], 'rb') as f:
                self.best_rl = pickle.load(f)
        else:
            self.best_rl = 0.0
        
        #* new get train dataloader
        self.prepare_train_dataloader()
        
    def get_tokenizer(self, model_dir):
        return AutoTokenizer.from_pretrained(self.init_model_dir, trust_remote_code=True)
    
    def prepare_train_dataloader(self):
        all_train_data_list = []
        with open(self.train_data_path, 'r') as file:
            for line in file.readlines():
                all_train_data_list.append(json.loads(line))
        
        bold_train_data_list = [item for item in all_train_data_list if 'neg_response' not in item or item['neg_response'] == None]
        orpo_train_data_list = [item for item in all_train_data_list if 'neg_response' in item and item['neg_response'] != None]

        bold_train_dataset = SingleR_Train_Dataset(bold_train_data_list, self.tokenizer)
        orpo_train_dataset = SingleR_Train_Dataset_Orpo(orpo_train_data_list, self.tokenizer)

        self.train_log_steps = 0
        if len(bold_train_dataset) > 0:
            self.bold_train_loader = DataLoader(dataset=bold_train_dataset,
                    shuffle=True,
                    batch_size=self.micro_batch_size,
                    collate_fn=collate_fn_Train, 
                    sampler=None
                )
            print(f"Len of bold_train_loader: {len(self.bold_train_loader)}")
            self.train_log_steps += math.ceil(
                len(self.bold_train_loader) / 10
            )
        else:
            self.bold_train_loader = None
            print(f"Len of bold_train_loader: {0}")

        if len(orpo_train_dataset) > 0:
            self.orpo_train_loader = DataLoader(dataset=orpo_train_dataset,
                    shuffle=True,
                    batch_size=self.micro_batch_size,
                    collate_fn=collate_fn_Train_orpo, 
                    sampler=None
                )
            print(f"Len of orpo_train_loader: {len(self.orpo_train_loader)}")
            self.train_log_steps += math.ceil(
                len(self.orpo_train_loader) / 10
            )
        else:
            self.orpo_train_loader = None
            print(f"Len of orpo_train_loader: {0}")


    
    def main_train(self, main_verbose = True):
        self.model.train()
        if main_verbose:
            print(f">>>>>>>>>>>>>>>>>LLM is loaded")
            print(">>>>>>>>>>>>>>>>>>Begin training")
        best_rl = self.best_rl
        # self.utility_evaluate()
        for epoch in range(self.epochs):
            train_loss_list = []
            ##### training code #####
            if self.rank ==0 and main_verbose:
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                print(f"Epoch {epoch}")
            self.model.train()
            global_step_idx = 0
            if self.bold_train_loader is not None:
                for idx, batch_text in enumerate(self.bold_train_loader):
                    global_step_idx += 1
                    if global_step_idx % 10 == 0 and main_verbose:
                        print(f"STEP {global_step_idx}/Total {len(self.train_dataloader)}")
                    record_loss = self.train_on_batch(batch_text=batch_text)
                    # print(record_loss)
                    
                    self.step(global_step_idx, record_loss)
                    train_loss_list.append(record_loss.mean().item())
                    torch.cuda.empty_cache()
                    if main_verbose and global_step_idx % self.train_log_steps == 0 and global_step_idx != 0:
                        print(f"===============================================")
                        print(f"[TRAINING SMAPLE]")
                        print(f"{epoch}th epoch, {global_step_idx}th batch: training loss: {np.mean(train_loss_list)}", flush=True)
                        print(f"===============================================")
                    if ((global_step_idx + 1) % self.n_eval_steps == 0) or ((global_step_idx + 1) == len(self.train_dataloader)):
                        train_loss_list = []
                        avg_rl, _, _ = self.utility_evaluate()
                        torch.cuda.empty_cache()
                        if main_verbose and avg_rl > best_rl:
                            print(f"STEP {global_step_idx}/EPOCH {epoch} SAVE MODEL")
                            best_rl = avg_rl
                            self.save_checkpoints(best_rl)

            if self.orpo_train_loader is not None:
                for idx, batch_text in enumerate(self.orpo_train_loader):
                    global_step_idx += 1
                    if global_step_idx % 10 == 0 and main_verbose:
                        print(f"STEP {global_step_idx}/Total {len(self.train_dataloader)}")
                    record_loss = self.train_on_batch_orpo(batch_text=batch_text)
                    # print(record_loss)
                    
                    self.step(global_step_idx, record_loss)
                    torch.cuda.empty_cache()
                    train_loss_list.append(record_loss.mean().item())
                    if main_verbose and global_step_idx % self.train_log_steps == 0 and global_step_idx != 0:
                        print(f"===============================================")
                        print(f"[TRAINING SMAPLE]")
                        print(f"{epoch}th epoch, {global_step_idx}th batch: training loss: {np.mean(train_loss_list)}", flush=True)
                        print(f"===============================================")
                    if ((global_step_idx + 1) % self.n_eval_steps == 0) or ((global_step_idx + 1) == len(self.train_dataloader)):
                        train_loss_list = []
                        avg_rl, _, _ = self.utility_evaluate()
                        if main_verbose and avg_rl > best_rl:
                            print(f"STEP {global_step_idx}/EPOCH {epoch} SAVE MODEL")
                            best_rl = avg_rl
                            self.save_checkpoints(best_rl)

        with open(self.best_rl_path, 'wb') as f:
            pickle.dump(best_rl, f)
    
    def train_on_batch_v2(self, batch_text):
        input_ids, labels, attn_masks = batch_text['input_ids'], batch_text['labels'], batch_text['attention_masks']
        # print(input_ids.shape)
        input_ids = input_ids.to(self.device)
        attn_masks = attn_masks.to(self.device)
        labels = labels.to(self.device)
        output = self.model(input_ids = input_ids, attention_mask=attn_masks, return_dict=True)
        logits = output['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(shift_logits.view(-1, shift_logits.size(2)), shift_labels.view(-1))
        loss = loss.view(input_ids.shape[0], -1)
        loss = loss.mean(1) 
        return loss

    def train_on_batch(self, batch_text):
        input_ids, labels, attn_masks = batch_text['input_ids'], batch_text['labels'], batch_text['attention_masks']
        # print(input_ids.shape)
        input_ids = input_ids.to(self.device)
        attn_masks = attn_masks.to(self.device)
        labels = labels.to(self.device)
        output = self.model(input_ids = input_ids, attention_mask=attn_masks, labels=labels)
        return output.loss


    def compute_logps(self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
        mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
        per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2, 
                                       index=(mask * chosen_inputs[:, 1:]).unsqueeze(2)).squeeze(2)
        return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)
    
    def train_on_batch_orpo(self, batch_text):
        input_ids, labels, attn_masks = batch_text['input_ids'], batch_text['labels'], batch_text['attention_masks']
        input_ids_neg, labels_neg, attn_masks_neg = batch_text['input_ids_neg'], batch_text['labels_neg'], batch_text['attention_masks_neg']
        prompt_masks = batch_text['prompt_masks'].to(self.device)

        input_ids, attn_masks, labels = input_ids.to(self.device), attn_masks.to(self.device), labels.to(self.device)
        input_ids_neg, attn_masks_neg, labels_neg = input_ids_neg.to(self.device), attn_masks_neg.to(self.device), labels_neg.to(self.device)
        neg_prompt_masks = batch_text['neg_prompt_masks'].to(self.device)

        outputs_pos = self.model(input_ids = input_ids, labels=labels, attention_mask=attn_masks, output_hidden_states = True)
        outputs_neg = self.model(input_ids = input_ids_neg, labels=labels_neg,  attention_mask=attn_masks_neg, output_hidden_states = True)
        

        pos_loss = outputs_pos.loss
        # calculate Log Probability
        pos_prob = self.compute_logps(
            prompt_attention_mask=prompt_masks,
            chosen_inputs=input_ids,
            chosen_attention_mask=attn_masks,
            logits=outputs_pos.logits
        )
        neg_prob = self.compute_logps(
            prompt_attention_mask=neg_prompt_masks,
            chosen_inputs=input_ids_neg,
            chosen_attention_mask=attn_masks_neg,
            logits=outputs_neg.logits
        )
        # Calculate log odds
        log_odds = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        self.alpha = 1.0
        # Calculate the Final Loss
        orpo_loss = torch.mean(pos_loss - self.alpha * ratio).to(dtype=torch.float32)
        return orpo_loss

class QLora_Llama_Finetune(DDP_QLora_trainer):
    def __init__(self, config):
        self.meter = Meter()

        #* add new parameter here
        super().__init__(**config)
        self.model_save_dir = os.path.join(config['model_save_dir'], f"BEST_MODEL")


def replace_config_file(config, file_path):
    if file_path == "": return
    with open(file_path, 'r') as file:
        data = hjson.load(file)
    for key, value in data.items():
        if key in config.keys():
            config[key] = value

if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ("true", "t", "yes", "y", "1", "True")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='model name')
    parser.add_argument("--use_DDP", type=str2bool, default="True", help='model name')
    parser.add_argument("--use_dp", type=str2bool, default="False", help='model name')
    parser.add_argument("--use_lora", type=str2bool, default="False", help='model name')
    parser.add_argument('--use_4_bits', type=str2bool, default="False",)
    parser.add_argument('--use_8_bits', type=str2bool, default="False",)


    parser.add_argument("--model_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--init_model_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--dataset_name", type=str, default="baize_dataset", help='model name')
    parser.add_argument("--dataset_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--train_data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--dev_data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--test_data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--collate_fn_name", type=str, default="collate_fn", help='local directory with model data')

    parser.add_argument("--max_generation_len", type=int, default=512, help='') # duplicate

    parser.add_argument("--model_save_dir", type=str, default="", help='local directory with model data')

    parser.add_argument("--freeze_embedding", type=str2bool, default="False", help='local directory with model data')
    parser.add_argument("--epoch", type=int, default=3, help='model name')
    parser.add_argument("--current_epoch", type=int, default=0, help='model name')
    # parser.add_argument("--eval_times", type=int, default=2, help='model name')
    parser.add_argument("--past_training_dir", type=str, default="", help='local directory with model data')


    parser.add_argument("--micro_batch_size", type=int, default=4, help='model name')
    parser.add_argument("--n_accumulation_steps", type=int, default=256, help='model name')
    parser.add_argument("--n_eval_steps", type=int, default=-1, help='model name')
    parser.add_argument('--lr', type=float, default=1e-4, help='as name')   
    parser.add_argument("--config_path", type=str, default="", help='model name')

    parser.add_argument("--is_single", type=str2bool, default=True, help='model name')

    parser.add_argument("--target_delta", type=float, default=1e-5, help='model name')
    parser.add_argument("--target_epsilon", type=float, default=8, help='model name')

    # Include DeepSpeed configuration arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    

    # Lora setting
    parser.add_argument('--lora_rank', type=int, default=64,
                    help='lora setting')
    parser.add_argument('--lora_alpha', type=int, default=16,
                    help='lora setting')
    parser.add_argument('--lora_droupout', type=float, default=0.0,
                    help='lora setting')


    parser.add_argument('--best_rl_path', type=str, default="", help='as name')   
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)
    
    # change to config param format
    config = {}

    # general
    config['model_dir'] = args.model_dir
    config['init_model_dir'] = args.init_model_dir
    config['optimizer_type'] = "adamw"
    config['lr'] = args.lr
    config['epochs'] = args.epoch
    config['current_epoch'] = args.current_epoch

    config['device_num'] = int(os.environ.get("WORLD_SIZE"))
    config['micro_batch_size'] = args.micro_batch_size
    config['n_accumulation_steps'] = args.n_accumulation_steps
    config['n_eval_steps'] = args.n_eval_steps
    # config['eval_times'] = args.eval_times

    config['dataset_name'] = args.dataset_name
    config['dataset_dir'] = args.dataset_dir
    config['train_data_path'] = args.train_data_path
    config['dev_data_path'] = args.dev_data_path
    config['test_data_path'] = args.test_data_path
    config['collate_fn_name'] = args.collate_fn_name
    config['max_generation_len'] = args.max_generation_len
    config['is_single'] = args.is_single
    config['model_save_dir'] = args.model_save_dir

    config['past_training_dir'] = args.past_training_dir

    
    # lora setting & quantization setting
    config['use_lora'] = args.use_lora
    config['lora_rank'] = args.lora_rank
    config['lora_alpha'] = args.lora_alpha
    config['lora_droupout'] = args.lora_droupout
    config['use_4_bits'] = args.use_4_bits
    config['use_8_bits'] = args.use_8_bits

    # DDP setting
    config['use_DDP'] = args.use_DDP

    config['best_rl_path'] = args.best_rl_path
    if args.use_DDP:
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config = hjson.load(f)
            # rewrite the deepspeed config file
            deepspeed_config['optimizer'] = {
                "type": config['optimizer_type'],
                "params": {
                    "lr": config['lr']
                }
            }
            deepspeed_config['train_micro_batch_size_per_gpu'] = config['micro_batch_size']
        config['deepspeed_config'] = deepspeed_config
        if args.use_dp:
            config['reduce_then_dp'] = False
    
    # dp setting
    config['dp'] = args.use_dp
    if args.use_dp:
        raise NotImplementedError("Not support DP now")
    else:
        replace_config_file(config, args.config_path)
        if args.use_lora:
            trainer = QLora_Llama_Finetune(config)
        else:
            trainer = Llama_Finetune(config)
    
    print(f"CONFIG: {config}")
    print(f"[Simplified CONFIG]: \n\t[Use DDP]: {args.use_DDP}\n\t[Use DP]: {args.use_dp}\n\t[Use LoRA]: " + \
          f"{args.use_lora}\n\t[Use 4 bits]: {args.use_4_bits}\n\t[Use 8 bits]: {args.use_8_bits}")
    trainer.main_train(main_verbose=args.local_rank == 0)

    save_epoch_model_path=os.path.join(config['model_save_dir'], f"MAIN_Epoch{config['current_epoch']}")
    trainer.save_model(save_epoch_model_path)
    print(f"Finish training at {TIME_STR} And save model to {save_epoch_model_path}")