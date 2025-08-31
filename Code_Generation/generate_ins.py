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

import time
TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

SYS_MSG= "You are a data creator and specialist tasked with generating a question based on the provided examples. " + \
        "Your task is to generate a new question similar with the provided examples. " + \
            "The question should be relevant to real-world scenarios and enhance the utility of the content for subsequent model training."
Input_prompt_template = "Come up with a series of tasks:\n\n## Example:\n### Instruction: {INS_1}\n\n## Example:\n### Instruction: {INS_2}\n\n## Example:\n### Instruction:"
def main(args):

    # 加载模型
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    print(f"device usage: {torch.cuda.cudart().cudaMemGetInfo(0)}")
    sampling_params = SamplingParams(n=args.gen_cnt, temperature=1.0, max_tokens=args.max_gen_len, seed=args.seed, top_p=0.7)

    tokenizer = AutoTokenizer.from_pretrained(args.init_model_dir)
    model = LLM(model=args.finetune_model_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.init_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len = 4096
              )

    test_dataList=[]
    with open(args.dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            test_dataList.append(data)

    all_seed_instruction = []
    for item in test_dataList:
        all_seed_instruction.append(f"{item['instruction']}\nPlease ensure that the function is named as {item['entry_point']}.")

    output_instruction_list = []
    with torch.no_grad():
        while True:
            random_seed_ins = random.sample(all_seed_instruction, 2)
            messages_question = [
                {
                    "role": "system",
                    "content": SYS_MSG,
                },
                {
                    "role": "user", "content": Input_prompt_template.format(
                        INS_1=random_seed_ins[0],
                        INS_2=random_seed_ins[1]
                    )
                },
            ]
            input_prompt = tokenizer.apply_chat_template(
                messages_question, add_generation_prompt=True, tokenize=False
            )

            output = model.generate(
                prompts=input_prompt,
                sampling_params=sampling_params
            )
            candidate_responses = [item.text for item in output[0].outputs]
            output_instruction_list.extend(candidate_responses)
            output_instruction_list = list(set(output_instruction_list))

            if len(output_instruction_list) >= args.target_dataset_size:
                break
            
            if len(output_instruction_list) % 1000 == 0:
                print(f"Generate {len(output_instruction_list)} instructions")

                # for item in candidate_responses:
                #     print(item)
                #     print("=====================================")
                # a+=1

            

    output_instruction_list = list(set(output_instruction_list))
    output_instruction_list = [{"instruction": item} for item in output_instruction_list]
    print("=====================================")
    print(f"[FINAL]")
    print("=====================================", flush=True)
    save_path = os.path.join(args.output_save_dir, f"gen_Q.json")
    with open(save_path, 'w') as f:
        for item in output_instruction_list:
            f.write(json.dumps(item) + "\n")
    print(f"Save to {save_path}")
    


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    def str2bool(s):
        return s.lower() in ("true", "t", "yes", "y", "1", "True")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='model name')
    parser.add_argument("--finetune_model_dir", type=str, default="", help='model name')
    parser.add_argument("--init_model_dir", type=str, default="", help='model name')

    parser.add_argument("--dataset_path", type=str, default="", help='model name')

    parser.add_argument("--gen_cnt", type=int, default=5, help='model name')

    parser.add_argument("--max_gen_len", type=int, default=512, help='model name')

    parser.add_argument("--target_dataset_size", type=int, default=1000, help='model name')

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help='model name')

    parser.add_argument("--output_save_dir", type=str, default="", help='model name')

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    main(args)