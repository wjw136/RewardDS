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
sys.path.append('../common')


from Dataset.finqa_dataset import SingleR_Dataset, collate_fn
import datetime
TIME= datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def main(args):

    # 加载模型
    device = torch.device("cuda:0")
    print(f"device usage: {torch.cuda.cudart().cudaMemGetInfo(0)}")
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_gen_len, seed=args.seed,
                                     logprobs = 1)

    tokenizer = AutoTokenizer.from_pretrained(args.initial_model_dir)
    finetune_model = LLM(model=args.finetune_model_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.initial_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len =2048
              )

    dp_finetune_model = LLM(model=args.dp_finetune_model_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.initial_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len =2048
              )
    
    init_model = LLM(model=args.initial_model_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.initial_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len =2048
              )


    test_dataset = SingleR_Dataset(tokenizer=tokenizer, mode = args.mode, train_data_path=args.train_data_path,
                                   test_data_path=args.test_data_path, dev_data_path=args.dev_data_path)  
    test_dataloader = DataLoader(dataset=test_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=collate_fn
                )
    
    # 评估
    dataList = []
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_dataloader)):
            questions, instructions, respones = batch_data['questions'], batch_data['instructions'], batch_data['responses']

            dp_finetune_outputs = dp_finetune_model.generate(
                prompts=instructions,
                # prompts= None,
                # prompt_token_ids = batch_data['instructions_ids'].numpy().tolist(),
                sampling_params=sampling_params
            )
            finetune_outputs = finetune_model.generate(
                prompts=instructions,
                # prompts= None,
                # prompt_token_ids = batch_data['instructions_ids'].numpy().tolist(),
                sampling_params=sampling_params
            )
            init_outputs = init_model.generate(
                prompts=instructions,
                # prompts= None,
                # prompt_token_ids = batch_data['instructions_ids'].numpy().tolist(),
                sampling_params=sampling_params
            )
        
            
            for question, dp_finetune_output, finetune_output, init_output, respone in zip(questions, dp_finetune_outputs, finetune_outputs, init_outputs, respones):
                dp_finetune_generated_text = tokenizer.decode(dp_finetune_output.outputs[0].token_ids, skip_special_tokens=True)
                finetune_generated_text = tokenizer.decode(finetune_output.outputs[0].token_ids, skip_special_tokens=True)
                init_generated_text = tokenizer.decode(init_output.outputs[0].token_ids, skip_special_tokens=True)

                dataList.append(
                    {
                        "question": question,
                        "dp_finetune_generated_text": dp_finetune_generated_text,
                        "finetune_generated_text": finetune_generated_text,
                        "init_generated_text": init_generated_text,
                        "respone": respone
                    }
                )

            if idx % 10 == 0:
                print("******************************")
                print("===sample===")
                print("dp_finetune_generated_text: ", dp_finetune_generated_text)
                print(f"finetune_generated_text: {finetune_generated_text}")
                print(f"init_generated_text: {init_generated_text}")
                print(f"respone: {respone}")
                print(f"******************************", flush=True)

    output_save_path = os.path.join(args.output_save_dir, f"{TIME}.json")
    print(f"Save to {output_save_path}")
    with open(output_save_path, "w") as f:
        for item in dataList:
            f.write(json.dumps(item) + "\n")
    


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
    parser.add_argument("--dp_finetune_model_dir", type=str, default="", help='model name')

    parser.add_argument("--batch_size", type=int, default=12, help='model name')

    parser.add_argument("--max_gen_len", type=int, default=512, help='model name')

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help='model name')

    parser.add_argument("--output_save_dir", type=str, default="", help='model name')

    parser.add_argument("--train_data_path", type=str, default="", help='model name')
    parser.add_argument("--test_data_path", type=str, default="", help='model name')
    parser.add_argument("--dev_data_path", type=str, default="", help='model name')
    parser.add_argument("--mode", type=str, default="train", help='model name')

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    main(args)