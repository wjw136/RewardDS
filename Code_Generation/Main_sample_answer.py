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
from Dataset.opc_dataset import SingleR_noBack_Dataset_noR, collate_fn_noBack_noR
import time
TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def main(args):

    # 加载模型
    device = torch.device("cuda:0")
    # print(f"device usage: {torch.cuda.cudart().cudaMemGetInfo(0)}")
    if not args.is_greedy:
        sampling_params = SamplingParams(n=args.gen_cnt, temperature=1.0, max_tokens=args.max_gen_len, seed=args.seed, top_p=0.7)
    else:
        sampling_params = SamplingParams(n=args.gen_cnt, temperature=0.0, max_tokens=args.max_gen_len, seed=args.seed, top_p=0.7)

    tokenizer = AutoTokenizer.from_pretrained(args.init_model_dir)
    model = LLM(model=args.finetune_model_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.init_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len = 4096
              )

    test_dataset = SingleR_noBack_Dataset_noR(test_data_path = args.dataset_path, tokenizer=tokenizer, mode = "test")
    test_dataloader = DataLoader(dataset=test_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=collate_fn_noBack_noR
                )
    test_dataList=[]
    with open(args.dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            test_dataList.append(data)

    output_data_list = []
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_dataloader)):
            input_prompts, instructions, entry_points = batch_data['input_prompts'], \
                        batch_data['instructions'], batch_data['entry_points']

            outputs = model.generate(
                prompts=input_prompts,
                sampling_params=sampling_params
            )
            for instruction, entry_point, output in zip(instructions, entry_points, outputs):
                
                candidate_responses = []
                for single_output in output.outputs:
                    candidate_responses.append(tokenizer.decode(single_output.token_ids, skip_special_tokens=True))

                output_data_list.append(
                    {
                        "instruction": instruction,
                        "entry_point": entry_point,
                        "candidate_responses": candidate_responses
                    }
                )

            if idx % 10 == 0:
                print("******************************")
                print(f"Processing {idx} / {len(test_dataloader)}")
                print(f"******************************", flush=True)


    print("=====================================")
    print(f"[FINAL]")
    print("=====================================", flush=True)
    save_path = os.path.join(args.output_save_dir, f"AllTrain_Generation_Epoch{args.epoch}.json")
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
    parser.add_argument("--dataset_path", type=str, default="", help='model name')
    parser.add_argument("--batch_size", type=int, default=5, help='model name')
    parser.add_argument("--max_gen_len", type=int, default=512, help='model name')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help='model name')
    parser.add_argument("--output_save_dir", type=str, default="", help='model name')

    parser.add_argument("--gen_cnt", type=int, default=6, help='model name')

    parser.add_argument("--epoch", type=int, default=42, help='model name')
    parser.add_argument("--is_greedy", type=str2bool, default=False, help='model name')
    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    main(args)