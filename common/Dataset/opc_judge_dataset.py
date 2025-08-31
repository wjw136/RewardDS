import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import config
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel, LoraConfig
import json
from torch.nn.utils.rnn import pad_sequence
import argparse
import importlib
import random
import numpy as np


####################################
####### 训练一个小的reward model ####
# 隐私数据质量评估
# level 1: Inital Q + DP A
# level 2: Inital Q + noDP A
# level 3: private Q + DP A
# level 4: private Q + noDP A
# level 5: private Q + private A
####################################


##############################
##### binary qualitifer ######
##############################
class Binary_TrainDataset(Dataset):
    def __init__(self, dataset_dir = "",
                    tokenizer=None, mode = "train",
                    train_data_path="", dev_data_path="", test_data_path=""):
        self.raw_data_list = []
        if mode == "train":
            data_path = os.path.join(dataset_dir, f"train.json") if train_data_path == "" else train_data_path
            with open(data_path, "r") as f:
                for line in f.readlines():
                    self.raw_data_list.append(json.loads(line))
        elif mode == "dev":
            self.data_path = os.path.join(dataset_dir, f"dev.json") if dev_data_path == "" else dev_data_path
            with open(self.data_path, "r") as f:
                for line in f.readlines():
                    self.raw_data_list.append(json.loads(line))
        elif mode == "test":
            self.data_path = os.path.join(dataset_dir, f"test.json") if test_data_path == "" else test_data_path
            with open(self.data_path, "r") as f:
                for line in f.readlines():
                    self.raw_data_list.append(json.loads(line))
        else:
            raise Exception("Invalid mode")
        
        if mode == 'dev':
            self.raw_data_list = self.raw_data_list
        self.tokenizer = tokenizer

        # process data:
        self.process_data_list = []
        for idx, item in enumerate(self.raw_data_list):
            
            item['init_generated_text'] = item['init_generated_text'].replace('<|im_end|>', '')
            item['respone'] = item['raw_response']
            item['dp_finetune_generated_text'] = item['dp_finetune_generated_text'].replace('<|im_end|>', '')
            item['finetune_generated_text'] = item['finetune_generated_text'].replace('<|im_end|>', '')
            instruction = f"{item['raw_instruction']}\nPlease ensure that the function is named as {item['entry_point']}."

            noTrain_sample = [
                {
                    "role": "system", "content": "You are a data quality evaluator. Please judge the quality of the following conversation."
                },
                {
                    "role": "user", "content": instruction
                },
                {
                    "role": "assistant", "content": item['init_generated_text']
                }
            ]

            gt_sample = [
                {
                    "role": "system", "content": "You are a data quality evaluator. Please judge the quality of the following conversation."
                },
                {
                    "role": "user", "content": instruction
                },
                {
                    "role": "assistant", "content": item['respone']
                }
            ]

            dp_sample = [
                {
                    "role": "system", "content": "You are a data quality evaluator. Please judge the quality of the following conversation."
                },
                {
                    "role": "user", "content": instruction
                },
                {
                    "role": "assistant", "content": item['dp_finetune_generated_text']
                }
            ]

            sn_sample = [
                {
                    "role": "system", "content": "You are a data quality evaluator. Please judge the quality of the following conversation."
                },
                {
                    "role": "user", "content": instruction
                },
                {
                    "role": "assistant", "content": item['finetune_generated_text']
                }
            ]

            noTrain_sample_prompt = self.tokenizer.apply_chat_template(
                noTrain_sample, add_generation_prompt=False, tokenize=False
            )
            dp_sample_prompt = self.tokenizer.apply_chat_template(
                dp_sample, add_generation_prompt=False, tokenize=False
            )
            sn_sample_prompt = self.tokenizer.apply_chat_template(
                sn_sample, add_generation_prompt=False, tokenize=False
            )
            gt_sample_prompt = self.tokenizer.apply_chat_template(
                gt_sample, add_generation_prompt=False, tokenize=False
            )

            self.process_data_list.extend(
                [
                    {
                        "idx": idx,
                        "chosen_sample": gt_sample_prompt,
                        "reject_sample": noTrain_sample_prompt,
                    },
                    {
                        "idx": idx,
                        "chosen_sample": gt_sample_prompt,
                        "reject_sample": sn_sample_prompt,
                    },
                    {
                        "idx": idx,
                        "chosen_sample": gt_sample_prompt,
                        "reject_sample": dp_sample_prompt,
                    },
                    {
                        "idx": idx,
                        "chosen_sample": sn_sample_prompt,
                        "reject_sample": noTrain_sample_prompt,
                    },
                    {
                        "idx": idx,
                        "chosen_sample": dp_sample_prompt,
                        "reject_sample": noTrain_sample_prompt,
                    }
                ]
            )

    def __getitem__(self, index):
        
        chosen_sample = self.process_data_list[index]['chosen_sample']
        reject_sample = self.process_data_list[index]['reject_sample']
        chosen_sample_ids = self.tokenizer.encode(chosen_sample, add_special_tokens=True)
        reject_sample_ids = self.tokenizer.encode(reject_sample, add_special_tokens=True)

        # response_ids = [1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 526, 8581, 491, 2531, 7492, 5478, 800, 393, 6602, 278, 289, 650, 1766, 798, 29915, 29879, 11509, 304, 7738, 10416, 9101, 29889, 4525, 5478, 800, 508, 6403, 805, 609, 1662, 5794, 470, 367, 23878, 515, 263, 3847, 29889, 450, 289, 650, 1766, 798, 13880, 10416, 9101, 29892, 3704, 2654, 10416, 9101, 29892, 4796, 10416, 9101, 29892, 322, 15284, 10376, 29889, 1932, 278, 289, 650, 1766, 798, 338, 9368, 304, 7738, 3307, 9045, 29891, 10416, 9101, 29892, 372, 508, 3275, 304, 385, 29747, 29892, 297, 1725, 1953, 29892, 322, 10767, 21219, 29889, 1670, 526, 3196, 4072, 310, 1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 29892, 3704, 1274, 1082, 590, 295, 3398, 454, 2679, 29747, 29892, 17168, 293, 590, 295, 3398, 454, 2679, 29747, 29892, 322, 590, 295, 397, 952, 572, 6288, 22898, 4871, 29889, 6479, 271, 358, 3987, 3160, 8950, 1228, 27580, 29892, 27310, 29220, 27580, 29892, 322, 289, 650, 1766, 798, 1301, 24389, 362, 29889, 2]
        return torch.tensor(chosen_sample_ids), torch.tensor(reject_sample_ids), chosen_sample, reject_sample
    
    def __len__(self):
        return len(self.process_data_list)
    


# collate_fn_xxx_eval/all
def lambda_collate_fn(batch, tokenizer=None):
    # 从批次中提取数据
    chosen_sample_ids = [item[0] for item in batch]
    reject_sample_ids = [item[1] for item in batch]
    chosen_sample = [item[2] for item in batch]
    reject_sample = [item[3] for item in batch]
    all_sample_ids = [ *chosen_sample_ids, *reject_sample_ids]

    # 填充输入文本数据
    chosen_sample_ids = pad_sequence(chosen_sample_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    reject_sample_ids = pad_sequence(reject_sample_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    all_sample_ids = pad_sequence(all_sample_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {'chosen_samples': chosen_sample,  'reject_samples': reject_sample, 
            'chosen_sample_ids': chosen_sample_ids, 'reject_sample_ids': reject_sample_ids,
            'all_sample_ids': all_sample_ids}




class EvaluateDataset_candidate(Dataset):
    def __init__(self, dataset_path = "", tokenizer=None ):
        self.raw_data_list = []
        self.tokenizer = tokenizer
        with open(dataset_path, "r") as f:
            for line in f.readlines():
                self.raw_data_list.append(json.loads(line))
        
        # process data:
        self.process_data_list = []
        for idx, item in enumerate(self.raw_data_list):
            instruction = f"{item['instruction']}\nPlease ensure that the function is named as {item['entry_point']}."
            candidate_samples = []
            candidate_responses = []
            for response in item['candidate_responses']:
                sample = [
                    {
                        "role": "system", "content": "You are a data quality evaluator. Please judge the quality of the following conversation."
                    },
                    {
                        "role": "user", "content": instruction
                    },
                    {
                        "role": "assistant", "content": response
                    }
                ]
                sample_prompt = self.tokenizer.apply_chat_template(
                    sample, add_generation_prompt=False, tokenize=False
                )
                candidate_samples.append(sample_prompt)
                candidate_responses.append(response)
            
            self.process_data_list.append(
                {
                    "instruction": item['instruction'],
                    "entry_point": item['entry_point'],
                    "candidate_samples": candidate_samples,
                    "candidate_responses": candidate_responses
                }
            )

    def __getitem__(self, index):

        instruction = self.process_data_list[index]['instruction']
        entry_point = self.process_data_list[index]['entry_point']
        candidate_samples = self.process_data_list[index]['candidate_samples']
        candidate_responses = self.process_data_list[index]['candidate_responses']

        # response_ids = [1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 526, 8581, 491, 2531, 7492, 5478, 800, 393, 6602, 278, 289, 650, 1766, 798, 29915, 29879, 11509, 304, 7738, 10416, 9101, 29889, 4525, 5478, 800, 508, 6403, 805, 609, 1662, 5794, 470, 367, 23878, 515, 263, 3847, 29889, 450, 289, 650, 1766, 798, 13880, 10416, 9101, 29892, 3704, 2654, 10416, 9101, 29892, 4796, 10416, 9101, 29892, 322, 15284, 10376, 29889, 1932, 278, 289, 650, 1766, 798, 338, 9368, 304, 7738, 3307, 9045, 29891, 10416, 9101, 29892, 372, 508, 3275, 304, 385, 29747, 29892, 297, 1725, 1953, 29892, 322, 10767, 21219, 29889, 1670, 526, 3196, 4072, 310, 1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 29892, 3704, 1274, 1082, 590, 295, 3398, 454, 2679, 29747, 29892, 17168, 293, 590, 295, 3398, 454, 2679, 29747, 29892, 322, 590, 295, 397, 952, 572, 6288, 22898, 4871, 29889, 6479, 271, 358, 3987, 3160, 8950, 1228, 27580, 29892, 27310, 29220, 27580, 29892, 322, 289, 650, 1766, 798, 1301, 24389, 362, 29889, 2]
        return instruction, entry_point, candidate_samples, candidate_responses
    
    def __len__(self):
        return len(self.process_data_list)
 
def lambda_collate_fn_candidate(batch, tokenizer=None):
    # 从批次中提取数据
    instruction = [item[0] for item in batch]
    entry_point = [item[1] for item in batch]
    candidate_samples = [item[2] for item in batch]
    candidate_responses = [item[3] for item in batch]

    return {'instruction': instruction, 'entry_point': entry_point, 'candidate_samples': candidate_samples, "candidate_responses": candidate_responses }