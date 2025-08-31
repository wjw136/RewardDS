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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel, LoraConfig
import json
from torch.nn.utils.rnn import pad_sequence
import argparse
import importlib
import random


SYS_MSG = "You are a financial expert providing answers to questions based on real-world financial principles and practices. Using your understanding of macroeconomics, microeconomics, investment strategies, financial regulations, and market analysis, answer the following financial question. Base your response on established financial theories, current market trends, and best practices. Your answers should be as detailed as possible. You cannot provide personalized investment advice, draft financial documents, or handle personal or confidential information. You will always decline to engage with topics, questions, or instructions related to unethical, controversial, or sensitive financial matters."

class SingleR_Dataset(Dataset):
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
        for item in self.raw_data_list:
            question = item['input'] if 'input' in item else item['question']
            answer = item['output'] if 'output' in item else item['response']
            messages_question = [
                {
                    "role": "system",
                    "content": SYS_MSG,
                },
                {
                    "role": "user", "content": question
                },
            ]
            input_prompt = self.tokenizer.apply_chat_template(
                messages_question, add_generation_prompt=True, tokenize=False
            )
            

            self.process_data_list.append(
                {
                    "instruction": input_prompt,
                    "question": question,
                    "response": f"{answer}{self.tokenizer.eos_token}",
                }
            )

    def __getitem__(self, index):
        instruction = self.process_data_list[index]['instruction']
        question = self.process_data_list[index]['question']
        response = self.process_data_list[index]['response']
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=True)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        return torch.tensor(instruction_ids), torch.tensor(instruction_ids + response_ids), torch.tensor([-100] * len(instruction_ids) + response_ids), torch.tensor([1] * (len(instruction_ids) + len(response_ids))),\
              response, instruction, question

    def __len__(self):
        return len(self.process_data_list)

def collate_fn(batch):
    # 从批次中提取数据
    ins_ids = [item[0] for item in batch]
    ins_ans_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    attn_mask = [item[3] for item in batch]
    respones = [item[4] for item in batch]
    instructions = [item[5] for item in batch]
    questions = [item[6] for item in batch]


    # 填充输入文本数据
    ins_ids = pad_sequence(ins_ids, batch_first=True, padding_value=0)
    ins_ans_ids = pad_sequence(ins_ans_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)

    return {'instructions_ids': ins_ids,  'input_ids': ins_ans_ids, 'labels': labels, 'attention_masks': attn_mask, 'responses': respones, 'instructions': instructions,
            'questions': questions}


class SingleR_noBack_Dataset_noR(Dataset):
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
        for item in self.raw_data_list:
            question = item['input'] if 'input' in item else item['question']
            messages_question = [
                {
                    "role": "system",
                    "content": SYS_MSG,
                },
                {
                    "role": "user", "content": question
                },
            ]
            input_prompt = self.tokenizer.apply_chat_template(
                messages_question, add_generation_prompt=True, tokenize=False
            )
            

            self.process_data_list.append(
                {
                    "input_prompt": input_prompt,
                    "question": question
                }
            )

    def __getitem__(self, index):
        input_prompt = self.process_data_list[index]['input_prompt']
        question = self.process_data_list[index]['question']
        
        return input_prompt, question

    def __len__(self):
        return len(self.process_data_list)

def collate_fn_noBack_noR(batch):
    # 从批次中提取数据
    input_prompts = [item[0] for item in batch]
    questions = [item[1] for item in batch]

    return {'input_prompts': input_prompts,
            'questions': questions}






class SingleR_Train_Dataset_Orpo(Dataset):
    def __init__(self, raw_data_list, tokenizer):
        self.tokenizer = tokenizer
        self.raw_data_list = raw_data_list

        # process data:
        self.process_data_list = []
        for item in self.raw_data_list:
            question = item['question'] if 'question' in item else item['input']
            response = item['response'][0] if type(item['response']) is list else item['response']
            neg_response = item['neg_response'][0] if type(item['neg_response']) is list else item['neg_response']
            response = response[0] if type(response) is list else response
            neg_response = neg_response[0] if type(neg_response) is list else neg_response

            messages_question = [
                {
                    "role": "system",
                    "content": SYS_MSG,
                },
                {
                    "role": "user", "content": question
                },
            ]
            input_prompt = self.tokenizer.apply_chat_template(
                messages_question, add_generation_prompt=True, tokenize=False
            )

            self.process_data_list.append(
                {
                    "question": question,
                    "instruction": input_prompt,
                    "response": f"{response}{self.tokenizer.eos_token}",
                    "neg_response": f"{neg_response}{self.tokenizer.eos_token}",
                }
            )

    def __getitem__(self, index):
        question = self.process_data_list[index]['question']
        instruction = self.process_data_list[index]['instruction']
        response = self.process_data_list[index]['response']
        neg_response = self.process_data_list[index]['neg_response']
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=True)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        neg_response_ids = self.tokenizer.encode(neg_response, add_special_tokens=False)

        # response_ids = [1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 526, 8581, 491, 2531, 7492, 5478, 800, 393, 6602, 278, 289, 650, 1766, 798, 29915, 29879, 11509, 304, 7738, 10416, 9101, 29889, 4525, 5478, 800, 508, 6403, 805, 609, 1662, 5794, 470, 367, 23878, 515, 263, 3847, 29889, 450, 289, 650, 1766, 798, 13880, 10416, 9101, 29892, 3704, 2654, 10416, 9101, 29892, 4796, 10416, 9101, 29892, 322, 15284, 10376, 29889, 1932, 278, 289, 650, 1766, 798, 338, 9368, 304, 7738, 3307, 9045, 29891, 10416, 9101, 29892, 372, 508, 3275, 304, 385, 29747, 29892, 297, 1725, 1953, 29892, 322, 10767, 21219, 29889, 1670, 526, 3196, 4072, 310, 1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 29892, 3704, 1274, 1082, 590, 295, 3398, 454, 2679, 29747, 29892, 17168, 293, 590, 295, 3398, 454, 2679, 29747, 29892, 322, 590, 295, 397, 952, 572, 6288, 22898, 4871, 29889, 6479, 271, 358, 3987, 3160, 8950, 1228, 27580, 29892, 27310, 29220, 27580, 29892, 322, 289, 650, 1766, 798, 1301, 24389, 362, 29889, 2]
        return torch.tensor(instruction_ids + response_ids), torch.tensor([-100] * len(instruction_ids) + response_ids), torch.tensor([1] * (len(instruction_ids) + len(response_ids))),\
        torch.tensor(instruction_ids + neg_response_ids), torch.tensor([-100] * len(instruction_ids) + neg_response_ids), torch.tensor([1] * (len(instruction_ids) + len(neg_response_ids))),\
        torch.tensor([1] * len(instruction_ids) + [0] * len(response_ids)), \
        torch.tensor([1] * len(instruction_ids) + [0] * len(neg_response_ids)), \
              question, instruction, response, neg_response

    def __len__(self):
        return len(self.process_data_list)


class SingleR_Train_Dataset(Dataset):
    def __init__(self, raw_data_list, tokenizer):
        self.tokenizer = tokenizer
        self.raw_data_list = raw_data_list

        # process data:
        self.process_data_list = []
        for item in self.raw_data_list:
            question = item['question'] if 'question' in item else item['input']
            response = item['response'][0] if type(item['response']) is list else item['response']
            response = response[0] if type(response) is list else response

            messages_question = [
                {
                    "role": "system",
                    "content": SYS_MSG,
                },
                {
                    "role": "user", "content": question
                },
            ]
            input_prompt = self.tokenizer.apply_chat_template(
                messages_question, add_generation_prompt=True, tokenize=False
            )

            self.process_data_list.append(
                {
                    "question": question,
                    "instruction": input_prompt,
                    "response": f"{response}{self.tokenizer.eos_token}",
                }
            )

    def __getitem__(self, index):
        question = self.process_data_list[index]['question']
        instruction = self.process_data_list[index]['instruction']
        response = self.process_data_list[index]['response']
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=True)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # response_ids = [1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 526, 8581, 491, 2531, 7492, 5478, 800, 393, 6602, 278, 289, 650, 1766, 798, 29915, 29879, 11509, 304, 7738, 10416, 9101, 29889, 4525, 5478, 800, 508, 6403, 805, 609, 1662, 5794, 470, 367, 23878, 515, 263, 3847, 29889, 450, 289, 650, 1766, 798, 13880, 10416, 9101, 29892, 3704, 2654, 10416, 9101, 29892, 4796, 10416, 9101, 29892, 322, 15284, 10376, 29889, 1932, 278, 289, 650, 1766, 798, 338, 9368, 304, 7738, 3307, 9045, 29891, 10416, 9101, 29892, 372, 508, 3275, 304, 385, 29747, 29892, 297, 1725, 1953, 29892, 322, 10767, 21219, 29889, 1670, 526, 3196, 4072, 310, 1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 29892, 3704, 1274, 1082, 590, 295, 3398, 454, 2679, 29747, 29892, 17168, 293, 590, 295, 3398, 454, 2679, 29747, 29892, 322, 590, 295, 397, 952, 572, 6288, 22898, 4871, 29889, 6479, 271, 358, 3987, 3160, 8950, 1228, 27580, 29892, 27310, 29220, 27580, 29892, 322, 289, 650, 1766, 798, 1301, 24389, 362, 29889, 2]
        return torch.tensor(instruction_ids), torch.tensor(instruction_ids + response_ids), \
            torch.tensor([-100] * len(instruction_ids) + response_ids), torch.tensor([1] * (len(instruction_ids) + len(response_ids))),\
              response, instruction, question

    def __len__(self):
        return len(self.process_data_list)


def collate_fn_Train(batch):
    # 从批次中提取数据
    ins_ids = [item[0] for item in batch]
    ins_ans_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    attn_mask = [item[3] for item in batch]
    respones = [item[4] for item in batch]
    instructions = [item[5] for item in batch]
    questions = [item[6] for item in batch]


    # 填充输入文本数据
    ins_ids = pad_sequence(ins_ids, batch_first=True, padding_value=0)
    ins_ans_ids = pad_sequence(ins_ans_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)

    return {'instructions_ids': ins_ids,  
            'input_ids': ins_ans_ids, 'labels': labels, 'attention_masks': attn_mask, 
            'responses': respones, 'instructions': instructions,
            "questions": questions}

def collate_fn_Train_orpo(batch):
    # 从批次中提取数据
    ins_ans_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    attn_mask = [item[2] for item in batch]
    ins_ans_ids_neg = [item[3] for item in batch]
    labels_neg = [item[4] for item in batch]
    attn_mask_neg = [item[5] for item in batch]
    prompt_masks = [item[6] for item in batch]
    neg_prompt_masks = [item[7] for item in batch]

    questions = [item[8] for item in batch]
    instructions = [item[9] for item in batch]
    respones = [item[10] for item in batch]
    neg_respones = [item[11] for item in batch]



    # 填充输入文本数据
    ins_ans_ids = pad_sequence(ins_ans_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
    ins_ans_ids_neg = pad_sequence(ins_ans_ids_neg, batch_first=True, padding_value=0)
    labels_neg = pad_sequence(labels_neg, batch_first=True, padding_value=-100)
    attn_mask_neg = pad_sequence(attn_mask_neg, batch_first=True, padding_value=0)
    prompt_masks = pad_sequence(prompt_masks, batch_first=True)
    neg_prompt_masks = pad_sequence(neg_prompt_masks, batch_first=True)


    return { 'input_ids': ins_ans_ids, 'labels': labels, 'attention_masks': attn_mask, 
            'input_ids_neg': ins_ans_ids_neg, 'labels_neg': labels_neg, 'attention_masks_neg': attn_mask_neg,
            'prompt_masks': prompt_masks, 'neg_prompt_masks': neg_prompt_masks,
            'responses': respones, 'neg_responses': neg_respones, 'instructions': instructions,
            "questions": questions}


###############
#! TextGrad
###############
SYS_MSG_grad="You are a smart language model that evaluates the training sample for the financial question answering task. " + \
    "Based on your understanding of basic and clinical science, financial knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, give the feedback for training sample. " + \
 "You should always provide evaluations in as much detail as possible. only evaluate existing solutions critically and give very concise feedback."

EVALUATION_SYS_TEMPLATE = """You are tasked with evaluating a chosen response by comparing it with a rejected response to a user query. Analyze the strengths and weaknesses of each response, step by step, and explain why one is chosen or rejected.

**User Query**:
{query}

**Chosen Response**:
{chosen_response}

**Rejected Response**:
{rejected_response}

**Do NOT generate a response to the query. Be concise.**"""

class Dataset_grad(Dataset):
    def __init__(self, dataset_path = "", tokenizer = None):
        self.raw_data_list = []
        with open(dataset_path, "r") as f:
            for line in f.readlines():
                self.raw_data_list.append(json.loads(line))
        
        self.tokenizer = tokenizer
        # process data:
        self.process_data_list = []
        for item in self.raw_data_list:
            question = item['input'] if 'input' in item else item['question']
            if 'rejected_response' in item:
                rejected_response = item['rejected_response'][0] if type(item['rejected_response']) is list else item['rejected_response']
            else:
                rejected_response = item['neg_response'][0] if type(item['neg_response']) is list else item['neg_response']
            messages_question = [
                {
                    "role": "system",
                    "content": SYS_MSG_grad,
                },
                {
                    "role": "user", "content": EVALUATION_SYS_TEMPLATE.format(
                        query=question,
                        chosen_response=item['response'][0] if type(item['response']) is list else item['response'],
                        rejected_response=rejected_response
                    )
                },
            ]
            input_prompt = self.tokenizer.apply_chat_template(
                messages_question, add_generation_prompt=True, tokenize=False
            )

            self.process_data_list.append(
                {
                    # "idx": item['idx'],
                    "instruction": input_prompt,
                    "question": question,
                }
            )

    def __getitem__(self, index):
        instruction = self.process_data_list[index]['instruction']
        question = self.process_data_list[index]['question']
        # response_ids = [1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 526, 8581, 491, 2531, 7492, 5478, 800, 393, 6602, 278, 289, 650, 1766, 798, 29915, 29879, 11509, 304, 7738, 10416, 9101, 29889, 4525, 5478, 800, 508, 6403, 805, 609, 1662, 5794, 470, 367, 23878, 515, 263, 3847, 29889, 450, 289, 650, 1766, 798, 13880, 10416, 9101, 29892, 3704, 2654, 10416, 9101, 29892, 4796, 10416, 9101, 29892, 322, 15284, 10376, 29889, 1932, 278, 289, 650, 1766, 798, 338, 9368, 304, 7738, 3307, 9045, 29891, 10416, 9101, 29892, 372, 508, 3275, 304, 385, 29747, 29892, 297, 1725, 1953, 29892, 322, 10767, 21219, 29889, 1670, 526, 3196, 4072, 310, 1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 29892, 3704, 1274, 1082, 590, 295, 3398, 454, 2679, 29747, 29892, 17168, 293, 590, 295, 3398, 454, 2679, 29747, 29892, 322, 590, 295, 397, 952, 572, 6288, 22898, 4871, 29889, 6479, 271, 358, 3987, 3160, 8950, 1228, 27580, 29892, 27310, 29220, 27580, 29892, 322, 289, 650, 1766, 798, 1301, 24389, 362, 29889, 2]
        return instruction, question

    def __len__(self):
        return len(self.process_data_list)


# new_variable_start_tag = '<IMPROVED_RESPONSE>'
# new_variable_end_tag = '</IMPROVED_RESPONSE>'

# SYS_MSG_optimizer="You are part of an optimization system that improves the response to the user query." + \
# "You will be asked to creatively and critically improve the response. You will receive some feedback, and use the feedback to improve the response. " + \
# "The feedback may be noisy, identify what is important and what is correct. " + \
# "This is very important: You MUST give your output by sending the improved response between {new_variable_start_tag} {{improved response}} {new_variable_end_tag} tags. " + \
# "The text you send between the tags will directly replace the response.\n\n"

# EVALUATION_SYS_TEMPLATE_optimizer = """You are tasked with improve the response to the user query according to the feedback. Here is the user query with response and feedback we got for the response. Please wrap your improved reponse between {new_variable_start_tag} {new_variable_end_tag}.

# **User Query**:
# {query}

# **Response**:
# {chosen_response}

# **Feedback**:
# {feedback}
# """

new_variable_start_tag = '<IMPROVED_RESPONSE>'
new_variable_end_tag = '</IMPROVED_RESPONSE>'

SYS_MSG_optimizer="You are part of an optimization system that improves the response to the user query." + \
"You will be asked to creatively and critically improve the response. You will receive some feedback, and use the feedback to improve the response. " + \
"The feedback may be noisy, identify what is important and what is correct. " + \
"This is very important: You MUST only output the improved response. " + \
"The text you send will directly replace the response.\n\n"

EVALUATION_SYS_TEMPLATE_optimizer = """You are tasked with improve the response to the user query according to the feedback. Here is the user query with response and feedback we got for the response. Please output your improved reponse.

**User Query**:
{query}

**Response**:
{chosen_response}

**Feedback**:
{feedback}

**Please improve the given response according to the feedback. Only output the improved response.**
"""

class Dataset_optimizer(Dataset):
    def __init__(self, dataset_path = "", tokenizer = None):
        self.raw_data_list = []
        with open(dataset_path, "r") as f:
            for line in f.readlines():
                self.raw_data_list.append(json.loads(line))
        
        self.tokenizer = tokenizer
        # process data:
        self.process_data_list = []
        for item in self.raw_data_list:
            question = item['input'] if 'input' in item else item['question']
            messages_question = [
                {
                    "role": "system",
                    "content": SYS_MSG_optimizer.format(
                        new_variable_start_tag=new_variable_start_tag,
                        new_variable_end_tag=new_variable_end_tag
                    ),
                },
                {
                    "role": "user", "content": EVALUATION_SYS_TEMPLATE_optimizer.format(
                        query=question,
                        chosen_response=item['response'][0] if type(item['response']) is list else item['response'],
                        feedback=item['feedback'],
                        new_variable_start_tag=new_variable_start_tag,
                        new_variable_end_tag=new_variable_end_tag
                    )
                },
            ]
            input_prompt = self.tokenizer.apply_chat_template(
                messages_question, add_generation_prompt=True, tokenize=False
            )

            self.process_data_list.append(
                {
                    # "idx": item['idx'],
                    "instruction": input_prompt,
                    "question": question,
                }
            )

    def __getitem__(self, index):
        instruction = self.process_data_list[index]['instruction']
        question = self.process_data_list[index]['question']
        # response_ids = [1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 526, 8581, 491, 2531, 7492, 5478, 800, 393, 6602, 278, 289, 650, 1766, 798, 29915, 29879, 11509, 304, 7738, 10416, 9101, 29889, 4525, 5478, 800, 508, 6403, 805, 609, 1662, 5794, 470, 367, 23878, 515, 263, 3847, 29889, 450, 289, 650, 1766, 798, 13880, 10416, 9101, 29892, 3704, 2654, 10416, 9101, 29892, 4796, 10416, 9101, 29892, 322, 15284, 10376, 29889, 1932, 278, 289, 650, 1766, 798, 338, 9368, 304, 7738, 3307, 9045, 29891, 10416, 9101, 29892, 372, 508, 3275, 304, 385, 29747, 29892, 297, 1725, 1953, 29892, 322, 10767, 21219, 29889, 1670, 526, 3196, 4072, 310, 1619, 295, 397, 952, 572, 6288, 22898, 456, 267, 29892, 3704, 1274, 1082, 590, 295, 3398, 454, 2679, 29747, 29892, 17168, 293, 590, 295, 3398, 454, 2679, 29747, 29892, 322, 590, 295, 397, 952, 572, 6288, 22898, 4871, 29889, 6479, 271, 358, 3987, 3160, 8950, 1228, 27580, 29892, 27310, 29220, 27580, 29892, 322, 289, 650, 1766, 798, 1301, 24389, 362, 29889, 2]
        return instruction, question

    def __len__(self):
        return len(self.process_data_list)


# collate_fn_xxx_eval/all
def collate_fn_noR(batch):
    instructions = [item[0] for item in batch]
    questions = [item[1] for item in batch]
    return {'instructions': instructions, 'questions': questions}