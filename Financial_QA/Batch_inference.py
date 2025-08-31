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

from Evaluate.evaluate_metrics import Meter

from Dataset.opc_dataset import SingleR_Dataset, collate_fn
import time
TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def main(args):

    # 加载模型
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    print(f"device usage: {torch.cuda.cudart().cudaMemGetInfo(0)}")
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_gen_len, seed=args.seed,
                                     logprobs = 1)

    tokenizer = AutoTokenizer.from_pretrained(args.initial_model_dir)
    model = LLM(model=args.finetune_model_dir, cpu_offload_gb=0, seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization = args.gpu_memory_utilization,
              tokenizer = args.initial_model_dir,
              skip_tokenizer_init = False,
              dtype = torch.float32,
              max_model_len =2048,
              enable_lora=args.use_lora,
              enforce_eager = False
              )
    if args.use_lora:
        loraRequest = LoRARequest(lora_name="adaptor", lora_int_id=0, lora_path=args.lora_path)


    test_dataset = SingleR_Dataset(test_data_path = args.test_data_path, tokenizer=tokenizer, mode = "test")
    test_dataloader = DataLoader(dataset=test_dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=collate_fn
                )
    
    # 评估
    output_data = []
    meter = Meter()
    with torch.no_grad():
        
        r1_list = []
        r2_list = []
        rl_list = []

        bleu_list = []
        ppl_list = []
        for idx, batch_data in enumerate(tqdm(test_dataloader)):
            instructions, questions, respones = batch_data['instructions'], batch_data['questions'], batch_data['responses']
            if not args.use_lora:
                outputs = model.generate(
                    prompts=instructions,
                    # prompts= None,
                    # prompt_token_ids = batch_data['instructions_ids'].numpy().tolist(),
                    sampling_params=sampling_params
                )
            else:
                outputs = model.generate(
                    prompts=instructions,
                    # prompt_token_ids = batch_data['instructions_ids'].numpy().tolist(),
                    sampling_params=sampling_params,
                    lora_request=loraRequest
                )

            
            for instruction, question, output, respone in zip(instructions, questions, outputs, respones):
                # print(instruction)
                output_len = len(output.outputs[0].logprobs)
                cumu_logprob = output.outputs[0].cumulative_logprob
                ppl = torch.exp(-torch.tensor(cumu_logprob)/output_len)
                ppl_list.append(ppl)
                # print(ppl)

                generated_text = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)
                r1, r2, rl = meter.compute_rouge(respone, generated_text)
                bleu = meter.compute_bleu(respone, generated_text)
                r1_list.append(r1)
                r2_list.append(r2)
                rl_list.append(rl)
                bleu_list.append(bleu)

                output_data.append(
                    {
                        "question": question,
                        "response": respone,
                        "generate_response": generated_text,
                        "rl": rl,
                    }
                )

                # print(f"response: {respone}")
                # print(f"generate text: {generated_text}", flush)


            if idx % 10 == 0:
                print("******************************")
                print(f"Processing {idx} / {len(test_dataloader)}")
                print(f"TMP AVG RL: {sum(rl_list) / len(rl_list)}")
                print(f"TMP AVG R1: {sum(r1_list) / len(r1_list)}")
                print(f"TMP AVG R2: {sum(r2_list) / len(r2_list)}")
                print(f"TMP AVG BLEU: {sum(bleu_list) / len(bleu_list)}")
                print(f"TMP AVG PPL: {sum(ppl_list) / len(ppl_list)}")
                print("===sample===")
                print(f"GENERATE: {generated_text}")
                print(f"REFERENCE: {respones[-1]}")
                print(f"[RL]: {rl}")
                print(f"---[R1]: {r1}, [R2]: {r2}")
                print(f"[BLEU]: {bleu}")
                print(f"[PPL]: {ppl}")
                print(f"******************************", flush=True)

    if args.save_dir != "":
        save_path = os.path.join(args.save_dir, f"output_{TIME}.json")
        with open(save_path, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item) + "\n")
        print(f"Save to {save_path}")

    print("=====================================")
    print(f"[FINAL]")
    print(f"AVG RL: {sum(rl_list) / len(rl_list)}")
    print(f"AVG R1: {sum(r1_list) / len(r1_list)}")
    print(f"AVG R2: {sum(r2_list) / len(r2_list)}")
    print(f"AVG BLEU: {sum(bleu_list) / len(bleu_list)}")
    print(f"AVG PPL: {sum(ppl_list) / len(ppl_list)}")
    print("=====================================", flush=True)
    torch.cuda.empty_cache()


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

    parser.add_argument("--test_data_path", type=str, default="", help='model name')

    parser.add_argument("--use_lora", type=str2bool, default=False, help='model name')
    parser.add_argument("--lora_path", type=str, default="", help='model name')

    parser.add_argument("--batch_size", type=int, default=5, help='model name')

    parser.add_argument("--max_gen_len", type=int, default=512, help='model name')

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help='model name')

    parser.add_argument("--save_dir", type=str, default="", help='model name')

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    main(args)