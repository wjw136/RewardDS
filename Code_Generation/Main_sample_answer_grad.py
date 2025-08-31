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

from Dataset.opc_dataset import Dataset_grad, collate_fn_noR, Dataset_optimizer, new_variable_start_tag, new_variable_end_tag
import time
import re
TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def main(args):

    # 加载模型
    device = torch.device("cuda:0")


    # ################################################
    # ########! text grad
    # ################################################
    sampling_params = SamplingParams(n=1, temperature=1.0, max_tokens=args.max_gen_len, seed=args.seed, top_p=0.7)

    tokenizer = AutoTokenizer.from_pretrained(args.init_model_dir)
    model = LLM(model=args.finetune_model_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.init_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len = 4096
              )

    test_dataset = Dataset_grad(dataset_path = args.dataset_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=collate_fn_noR
                )
    test_dataList=[]
    with open(args.dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            test_dataList.append(data)

    output_data_list = []
    with torch.no_grad():   
        for idx, batch_data in enumerate(tqdm(test_dataloader)):
            input_prompts, instructions = batch_data['input_prompts'], batch_data['instructions']

            outputs = model.generate(
                prompts=input_prompts,
                sampling_params=sampling_params
            )
            for sub_idx, (input_prompt, instruction, output) in enumerate(zip(input_prompts, instructions, outputs)):

                output_data_list.append(
                    {
                        "instruction": instruction,
                        "response": test_dataList[idx * args.batch_size + sub_idx]['response'],
                        "feedback": tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)
                    }
                )

            if idx % 10 == 0:
                print("******************************")
                print(f"[FeedBack] Processing {idx} / {len(test_dataloader)}")
                print(f"******************************", flush=True)

    print("=====================================")
    print(f"[FeedBack FINAL]")
    print("=====================================", flush=True)
    save_path = os.path.join(args.output_save_dir, f"AllTrain_FeedBack_Epoch{args.epoch}.json")
    with open(save_path, 'w') as f:
        for item in output_data_list:
            f.write(json.dumps(item) + "\n")
    print(f"Save to {save_path}")
    args.dataset_path = save_path

    ################################################
    ########! text grad
    ################################################
    del model
    del tokenizer
    torch.cuda.empty_cache()

    sampling_params = SamplingParams(n=args.gen_cnt, temperature=1.0, max_tokens=args.max_gen_len, seed=args.seed, top_p=0.7)

    tokenizer = AutoTokenizer.from_pretrained(args.init_opti_dir)
    model = LLM(model=args.finetune_opti_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.init_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len = 4096
              )

    test_dataset = Dataset_optimizer(dataset_path = args.dataset_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset=test_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=collate_fn_noR
                )
    test_dataList=[]
    with open(args.dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            test_dataList.append(data)

    pattern = re.compile(f"{new_variable_start_tag}(.*){new_variable_end_tag}")
    output_data_list = []
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_dataloader)):
            input_prompts, instructions = batch_data['input_prompts'], batch_data['instructions']

            outputs = model.generate(
                prompts=input_prompts,
                sampling_params=sampling_params
            )
            for sub_idx, (input_prompt, instruction, output) in enumerate(zip(input_prompts, instructions, outputs)):
                
                # print(instruction)
                candidate_responses = []
                for single_output in output.outputs:
                    if single_output.token_ids[-1] == tokenizer.eos_token_id:
                        candidate_res = tokenizer.decode(single_output.token_ids, skip_special_tokens=True)
                        # print(len(single_output.token_ids))
                        # print(f">>> {candidate_res}")
                        # print(f">>> {candidate_res}")
                        # if len(pattern.findall(candidate_res)) > 0:
                            # candidate_responses.append(pattern.findall(candidate_res)[0])
                        candidate_responses.append(candidate_res)
                # print("==================================")
                
                if len(candidate_responses) == 0:
                    print(f"Error: {instruction}")
                    continue
                output_data_list.append(
                    {
                        "instruction": instruction,
                        "candidate_responses": candidate_responses
                    }
                )

            if idx % 10 == 0:
                print("******************************")
                print(f"[Revise] Processing {idx} / {len(test_dataloader)}")
                print(f"******************************", flush=True)

    print("=====================================")
    print(f"[Revise FINAL]")
    print("=====================================", flush=True)
    save_path = os.path.join(args.output_save_dir, f"AllTrain_Revise_Epoch{args.epoch}.json")
    with open(save_path, 'w') as f:
        for item in output_data_list:
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
    parser.add_argument("--finetune_opti_dir", type=str, default="", help='model name')
    parser.add_argument("--init_opti_dir", type=str, default="", help='model name')
    parser.add_argument("--dataset_path", type=str, default="", help='model name')
    parser.add_argument("--batch_size", type=int, default=5, help='model name')
    parser.add_argument("--max_gen_len", type=int, default=512, help='model name')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help='model name')
    parser.add_argument("--output_save_dir", type=str, default="", help='model name')

    parser.add_argument("--gen_cnt", type=int, default=6, help='model name')

    parser.add_argument("--epoch", type=int, default=1, help='model name')
    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    main(args)