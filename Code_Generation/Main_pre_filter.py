from transformers import AutoTokenizer
import argparse
import torch
import random
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel, LoraConfig
import importlib
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
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
from transformers import Qwen2ForSequenceClassification


import datetime
TIME= datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
import pickle
import numpy as np

def main(args):

    data_list = []
    with open(args.data_path, 'r') as file:
        for line in file.readlines():
            data_list.append(json.loads(line))
    
    process_data_list = []
    for idx, item in enumerate(data_list):
        if idx % 100 == 0:
            print(f">>> {idx}/{len(data_list)}")
        all_response = item['candidate_responses_with_logit']
        
        # 根据tuple的第一个元素排序: score
        sorted_by_first = sorted(enumerate(all_response), key=lambda x: x[1][1])
        rank_by_first = [sorted_by_first.index((i, val)) + 1 for i, val in enumerate(all_response)]

        # 根据tuple的第二个元素排序: ppl
        sorted_by_second = sorted(enumerate(all_response), key=lambda x: x[1][2], reverse=True)
        rank_by_second = [sorted_by_second.index((i, val)) + 1 for i, val in enumerate(all_response)]

        rank = [r_first * args.alpha + r_second * (1-args.alpha) 
                for r_first, r_second in zip(rank_by_first, rank_by_second)]
        # For Pos Response
        pos_response = all_response[np.argmax(rank)]
        neg_response = all_response[np.argmin(rank)]

        process_data_list.append(
            {
                "instruction": item['instruction'],
                "response": pos_response[0],
                "rejected_response": neg_response[0],
                "rw_score": pos_response[1]
            }
        )

    print(f"CNT: {len(data_list)}")

    sort_process_data_list = sorted(process_data_list, key=lambda x: x['rw_score'], reverse=True)
    select_cnt = int(len(sort_process_data_list) / args.total_split * 1)
    pick_data_list = sort_process_data_list[0:select_cnt]

    print(f"Select CNT: {len(pick_data_list)}")

    save_path=f"{args.output_save_dir}/Filter_revise_data_Epoch1.json"
    with open(save_path, 'w') as f:
        for item in pick_data_list:
            f.write(json.dumps(item)+'\n')
    
    save_path=f"{args.output_save_dir}/Revise_data_Epoch1.json"
    with open(save_path, 'w') as f:
        for item in process_data_list:
            f.write(json.dumps(item)+'\n')

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    def str2bool(s):
        return s.lower() in ("true", "t", "yes", "y", "1", "True")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='model name')

    parser.add_argument("--output_save_dir", type=str, default="", help='model name')

    parser.add_argument("--data_path", type=str, default="", help='model name')

    parser.add_argument("--total_split", type=int, default=0, help='model name')

    parser.add_argument("--alpha", type=float, default=1, help='model name')
    parser.add_argument("--diff_thres", type=float, default=100000, help='model name')

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    main(args)