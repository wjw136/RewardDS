import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../common')
# print(sys.path)
import config
from training_interface import DP_DDP_trainer, DDP_trainer, DDP_QLora_trainer
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
from PrivLM_Bench.eval.metrics import Meter
import deepspeed
import hjson
import random
from torch.utils.data.distributed import DistributedSampler
from deepspeed import comm as dist 
import numpy as np
from opc_utils.eval_pass import single_calculate_accuracy, calculate_accuracy
import pickle
import time
TIME_STR = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

class Llama_Finetune(DDP_trainer):
    def __init__(self, config):
        self.meter = Meter()
        self.tmp_dir = config['tmp_dir']
        self.init_model_dir = config['init_model_dir']

        #* add new parameter here
        super().__init__(**config)

        self.model_save_dir = os.path.join(config['model_save_dir'], f"BEST_MODEL")
        self.best_acc_path = config['best_acc_path']
        if os.path.exists(config['best_acc_path']):
            with open(config['best_acc_path'], 'rb') as f:
                self.best_acc = pickle.load(f)
        else:
            self.best_acc = 0.0
    
    def get_tokenizer(self, model_dir):
        return AutoTokenizer.from_pretrained(self.init_model_dir, trust_remote_code=True)

    def utility_evaluate(self):
        self.model.eval()
        acc_list = []
        with torch.no_grad():
            for idx, batch_text in enumerate(self.dev_dataloader):
                if self.rank == 0 and idx % 10 == 0:
                    print(f"[DEV]: STEP {idx}/Total {len(self.dev_dataloader)}")

                ins_ids, instruction, answers = batch_text['instructions_ids'], batch_text['instructions'], \
                    batch_text['responses']
                ins_ids = ins_ids.to(self.device)
                input_length = ins_ids.shape[1]

                output = self.model.generate(inputs=ins_ids, max_new_tokens = self.max_generation_len, do_sample=False,
                                            top_p = 1.0, temperature=None, top_k=None, num_return_sequences=1,
                                            pad_token_id=self.tokenizer.eos_token_id)
                torch.cuda.empty_cache()

                generated_text = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=False)

                if single_calculate_accuracy(batch_text["seq_ids"][0], generated_text, 
                                             batch_text['entry_points'][0], batch_text['testcases'][0], self.tmp_dir):
                    acc_list.append(1)
                else:
                    acc_list.append(0)
                
                torch.cuda.empty_cache()

                if self.rank==0 and idx % 10 == 0:
                    print("*********************************")
                    print("[Dev sample]")
                    print(f"instruction: {instruction}")
                    print(f"GENERATE: {generated_text}")
                    print(f"REFERENCE: {answers}")
                    print(f"[Pass@1]: {acc_list[-1]}")
                    print(f"[TMP Avg Pass@1]: {sum(acc_list) / len(acc_list)}", flush = True)
                    print("*********************************")

            acc_tensor = torch.tensor(acc_list).to(self.device).to(torch.float)

            gathered_acc = [torch.zeros_like(acc_tensor).to(self.device) for _ in range(self.world_size)]
            dist.all_gather(gathered_acc, acc_tensor)

            if self.rank == 0:
                avg_acc = torch.cat(gathered_acc).mean().cpu().numpy()
                print("=================================")
                print("[FINAL Dev result]")
                print(f"ACC: {avg_acc}", flush=True)
                print("=================================")
            else:
                avg_acc = -1
            # 同步
            dist.barrier()
        self.model.train()
        torch.cuda.empty_cache()

        return avg_acc

    def main_train(self, main_verbose = True):

        self.model.train()
        if main_verbose:
            print(f">>>>>>>>>>>>>>>>>LLM is loaded")
            print(">>>>>>>>>>>>>>>>>>Begin training")
        best_acc = self.best_acc
        # self.utility_evaluate()
        for epoch in range(self.epochs):
            train_loss_list = []
            ##### training code #####
            if self.rank ==0 and main_verbose:
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                print(f"Epoch {epoch}")
            self.model.train()
            for idx, batch_text in enumerate(self.train_dataloader):
                if idx % 10 == 0 and main_verbose:
                    print(f"STEP {idx}/Total {len(self.train_dataloader)}")
                record_loss = self.train_on_batch(batch_text=batch_text)
                # print(record_loss)
                
                self.step(idx, record_loss)
                torch.cuda.empty_cache()
                train_loss_list.append(record_loss.mean().item())
                if main_verbose and idx % self.train_log_steps == 0 and idx != 0:
                    print(f"===============================================")
                    print(f"[TRAINING SMAPLE]")
                    print(f"{epoch}th epoch, {idx}th batch: training loss: {np.mean(train_loss_list)}", flush=True)
                    print(f"===============================================")
                if ((idx + 1) % self.n_eval_steps == 0) or ((idx + 1) == len(self.train_dataloader)):
                    train_loss_list = []
                    avg_acc = self.utility_evaluate()
                    if main_verbose and avg_acc > best_acc:
                        print(f"STEP {idx}/EPOCH {epoch} SAVE MODEL")
                        best_acc = avg_acc
                        self.save_checkpoints(avg_acc)

        with open(self.best_acc_path, 'wb') as f:
            pickle.dump(best_acc, f)


    def save_checkpoints(self, best_acc):
        self.model.save_pretrained(self.model_save_dir)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(f"best Pass@1: {best_acc}")
        print(f"[SAVE] Best model to {self.model_save_dir}")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")




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

    parser.add_argument('--best_acc_path', type=str, default="", help='as name')   

    parser.add_argument("--tmp_dir", type=str, default="", help='model name')

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

    config['tmp_dir'] = args.tmp_dir

    
    # lora setting & quantization setting
    config['use_lora'] = args.use_lora
    config['lora_rank'] = args.lora_rank
    config['lora_alpha'] = args.lora_alpha
    config['lora_droupout'] = args.lora_droupout
    config['use_4_bits'] = args.use_4_bits
    config['use_8_bits'] = args.use_8_bits

    # DDP setting
    config['use_DDP'] = args.use_DDP

    config['best_acc_path'] = args.best_acc_path
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
        raise ValueError("Not implemented yet")
    else:
        replace_config_file(config, args.config_path)
        if args.use_lora:
            pass
        else:
            trainer = Llama_Finetune(config)
    
    print(f"CONFIG: {config}")
    print(f"[Simplified CONFIG]: \n\t[Use DDP]: {args.use_DDP}\n\t[Use DP]: {args.use_dp}\n\t[Use LoRA]: " + \
          f"{args.use_lora}\n\t[Use 4 bits]: {args.use_4_bits}\n\t[Use 8 bits]: {args.use_8_bits}")
    trainer.main_train(main_verbose=args.local_rank == 0)

    save_epoch_model_path=os.path.join(config['model_save_dir'], f"MAIN_Epoch{config['current_epoch']}")
    trainer.save_model(save_epoch_model_path)
    print(f"Finish training at {TIME_STR} And save model to {save_epoch_model_path}")