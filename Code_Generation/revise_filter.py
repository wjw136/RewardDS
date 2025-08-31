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

    previous_data_list = []
    with open(args.previous_data_path, 'r') as file:
        for line in file.readlines():
            previous_data_list.append(json.loads(line))
    sort_previous_data_list = sorted(previous_data_list, key=lambda x: x['rw_score'], reverse=True)
    threshold = float(sort_previous_data_list[-1]['rw_score'])
    print(f"Threshold: {threshold}")

    has_neg_cnt = 0
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
                "rw_score": float(pos_response[1]),
            }
        )

    print(f"CNT: {len(data_list)}/Neg CNT: {has_neg_cnt}")

    sort_process_data_list = sorted(process_data_list, key=lambda x: x['rw_score'], reverse=True)
    pick_data_list = [item for item in sort_process_data_list if (item['rw_score'] * args.ratio_to_first) >= threshold]

    # print(pick_data_list)

    print(f"Select CNT: {len(pick_data_list)}")

    if args.diff_thres == -1.0:
        save_data_list = pick_data_list + previous_data_list
    else:
        save_data_list = pick_data_list

    save_path=f"{args.output_save_dir}/Filter_revise_data_Epoch{args.epoch}.json"
    with open(save_path, 'w') as f:
        for item in save_data_list:
            f.write(json.dumps(item)+'\n')
    print(f"Save to {save_path}")
    

    save_path=f"{args.output_save_dir}/Revise_data_Epoch{args.epoch}.json"
    with open(save_path, 'w') as f:
        for item in process_data_list:
            f.write(json.dumps(item)+'\n')
    print(f"Save to {save_path}")


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

    parser.add_argument("--previous_data_path", type=str, default="", help='model name')

    parser.add_argument("--epoch", type=int, default=0, help='model name')

    parser.add_argument("--alpha", type=float, default=0.5, help='model name')
    parser.add_argument("--diff_thres", type=float, default=0.0, help='model name')

    parser.add_argument("--ratio_to_first", type=float, default=0.7, help='model name')

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)
    main(args)