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
from transformers import Qwen2ForSequenceClassification, AutoModelForCausalLM
import numpy as np


import datetime
TIME= datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")



class EvaluateDataset_candidate_moment(Dataset):
    def __init__(self, dataset_path = "", tokenizer=None ):
        self.raw_data_list = []
        self.tokenizer = tokenizer
        with open(dataset_path, "r") as f:
            for line in f.readlines():
                self.raw_data_list.append(json.loads(line))
        
       # process data:
        self.process_data_list = []
        for idx, item in enumerate(self.raw_data_list):
            question = item['input'] if 'input' in item else item['question']
            all_resposenes = item['candidate_responses']
            if len(all_resposenes) == 0:
                continue

            candidate_samples = []
            candidate_responses = []

            # print(item['response'])
            # print(item['candidate_responses'])
            #! 顺序问题
            for response in all_resposenes:
                sample = [
                    {
                        "role": "system", "content": "You are a data quality evaluator. Please judge the quality of the following conversation."
                    },
                    {
                        "role": "user", "content": question
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
                    "question": question,
                    "candidate_samples": candidate_samples,
                    "candidate_responses": candidate_responses
                }
            )

    def __getitem__(self, index):

        question = self.process_data_list[index]['question']
        candidate_samples = self.process_data_list[index]['candidate_samples']
        candidate_responses = self.process_data_list[index]['candidate_responses']

        # response_ids = [1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 526, 8581, 491, 2531, 7492, 5478, 800, 393, 6602, 278, 289, 650, 1766, 798, 29915, 29879, 11509, 304, 7738, 10416, 9101, 29889, 4525, 5478, 800, 508, 6403, 805, 609, 1662, 5794, 470, 367, 23878, 515, 263, 3847, 29889, 450, 289, 650, 1766, 798, 13880, 10416, 9101, 29892, 3704, 2654, 10416, 9101, 29892, 4796, 10416, 9101, 29892, 322, 15284, 10376, 29889, 1932, 278, 289, 650, 1766, 798, 338, 9368, 304, 7738, 3307, 9045, 29891, 10416, 9101, 29892, 372, 508, 3275, 304, 385, 29747, 29892, 297, 1725, 1953, 29892, 322, 10767, 21219, 29889, 1670, 526, 3196, 4072, 310, 1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 29892, 3704, 1274, 1082, 590, 295, 3398, 454, 2679, 29747, 29892, 17168, 293, 590, 295, 3398, 454, 2679, 29747, 29892, 322, 590, 295, 397, 952, 572, 6288, 22898, 4871, 29889, 6479, 271, 358, 3987, 3160, 8950, 1228, 27580, 29892, 27310, 29220, 27580, 29892, 322, 289, 650, 1766, 798, 1301, 24389, 362, 29889, 2]
        return question, candidate_samples, candidate_responses
    
    def __len__(self):
        return len(self.process_data_list)


def lambda_collate_fn_candidate_moment(batch, tokenizer=None):
    # 从批次中提取数据
    question = [item[0] for item in batch]
    candidate_samples = [item[1] for item in batch]
    candidate_responses = [item[2] for item in batch]
    
    return { 'question': question, 'candidate_samples': candidate_samples, "candidate_responses": candidate_responses}

import torch
import torch.nn.functional as F

SYS_MSG = "You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following medical question. Base your answer on the current and standard practices referenced in medical guidelines. You should always provides responses in as much detail as possible. You can not help with doctor appointments and will never ask personal information. You always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues."

def cal_ppl(model, tokenizer, device, question, response):
    # 将问题和回答分开，确保回答部分作为目标
    question_text = [
                {
                    "role": "system",
                    "content": SYS_MSG,
                },
                {
                    "role": "user", "content": question
                },
    ]
    question_text = tokenizer.apply_chat_template(
        question_text, add_generation_prompt=True, tokenize=False
    ) # 问题部分
    target_text = response  # 目标为回答部分
    
    # 使用 tokenizer 对文本进行编码
    question_ids = tokenizer(question_text, return_tensors='pt').to(device)['input_ids']
    target_ids = tokenizer(target_text, return_tensors='pt').to(device)['input_ids']
    all_ids = tokenizer(question_text + target_text, return_tensors='pt').to(device)['input_ids']
    
    input_ids = all_ids[:, :-1]
    label_ids = all_ids[:, 1:]
    attn_mask = torch.tensor([1] * (question_ids.shape[1] - 1) + [0] * (target_ids.shape[1])).to(device).unsqueeze(0)
    # attn_mask = torch.tensor([1] * (question_ids.shape[1]) + [0] * (target_ids.shape[1] -1)).to(device).unsqueeze(0)

    new_label_ids = torch.concat([torch.tensor([-100] * question_ids.shape[1]).unsqueeze(0).to(device), target_ids], dim=1).to(torch.int64)
    # 获取模型的输出
    with torch.no_grad():
        # 模型的输入是问题部分，目标是回答部分
        outputs = model(all_ids, labels=new_label_ids)
    
    # 计算负对数似然
    loss = outputs.loss  # 计算的损失即是负对数似然
    ppl = torch.exp(loss)  # 困惑度是负对数似然的指数
    return ppl.item()  # 返回困惑度作为标量


def main(args):
    device = torch.device("cuda:0")
    # 加载模型
    model = Qwen2ForSequenceClassification.from_pretrained(args.finetune_model_dir, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.initial_model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id

    ppl_model = AutoModelForCausalLM.from_pretrained(args.ppl_model_dir, device_map=device)

    test_dataset = EvaluateDataset_candidate_moment(dataset_path = args.dataset_path, tokenizer=tokenizer)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                    shuffle=False,
                    batch_size=args.batch_size,
                    collate_fn=lambda_collate_fn_candidate_moment,
                )

    # load dataset 
    test_datalist = []
    with open(args.dataset_path, 'r') as f:
        for line in f:
            test_datalist.append(json.loads(line))
    print(f"test dataset size: {len(test_datalist)}")

    with torch.no_grad():
        torch.cuda.empty_cache()
        all_processed_data = []
        for idx, batch in enumerate(tqdm(test_dataloader)):
            questions = batch['question']
            candidate_samples = batch['candidate_samples']
            candidate_responses= batch['candidate_responses']
            
            for i in range(len(candidate_samples)):
                
                input_sample_ids = tokenizer.batch_encode_plus(candidate_samples[i], padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
                output_logits = model(input_ids = input_sample_ids).logits.unsqueeze(-1)

                response_logits = []
                for j in range(output_logits.shape[0]):
                    response_logits.append(output_logits[j].item())
                
                response_ppls = []
                for j in range(len(candidate_responses[i])):
                    response_ppl = cal_ppl(ppl_model, tokenizer, device, questions[i], candidate_responses[i][j])
                    response_ppls.append(response_ppl)


                candidate_responses_with_logit = [(item, response_logits[idx], response_ppls[idx]) for idx, item in enumerate(candidate_responses[i])]
                
                all_processed_data.append(
                    {
                    'question': questions[i],
                    'candidate_responses_with_logit': candidate_responses_with_logit
                    }
                )
            
            if idx % 100 == 0:
                print(f"processed {idx} / TOTAL {len(test_dataloader)}", flush=True)
                print(all_processed_data[-1], flush=True)

    output_save_path = os.path.join(args.output_save_dir, f"AllTrain_Judge_Epoch{args.epoch}.json")
    with open(output_save_path, 'w') as f:
        for item in all_processed_data:
            f.write(json.dumps(item) + '\n')
    print(f"output save to {output_save_path}")

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    def str2bool(s):
        return s.lower() in ("true", "t", "yes", "y", "1", "True")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='model name')

    parser.add_argument("--initial_model_dir", type=str, default="", help='model name')
    parser.add_argument("--finetune_model_dir", type=str, default="", help='model name')


    parser.add_argument("--ppl_model_dir", type=str, default="", help='model name')

    parser.add_argument("--batch_size", type=int, default=12, help='model name')

    parser.add_argument("--output_save_dir", type=str, default="", help='model name')

    parser.add_argument("--dataset_path", type=str, default="", help='model name')
    parser.add_argument("--epoch", type=int, default=12, help='model name')

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    main(args)