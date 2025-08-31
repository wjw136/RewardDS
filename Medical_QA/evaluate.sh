set -e

Qwen2d5_7B_dir=""




#############################################################################
# Step1. Evaluate the fine-tuned LLM 
#############################################################################
finetune_dir=""
CUDA_VISIBLE_DEVICES=2 python Batch_inference.py \
                --test_data_path "" \
                --initial_model_dir "$Qwen2d5_7B_dir" \
                --finetune_model_dir "$finetune_dir" \
                --batch_size 12 \
                --gpu_memory_utilization 0.8
exit 0

