set -e

Qwen2d5_0d5B_dir=""

##############################################################
# Step1: Direct Finetune Proxy model without DP
##############################################################
cd ../common/PrivLM_Bench/examples
deepspeed  \
    --include localhost:0 \
    --master_port 29502  ./llama_finetune.py \
    --model_dir "$Qwen2d5_0d5B_dir" \
    --dataset_name opc_dataset.SingleR_Dataset \
    --collate_fn_name "collate_fn" \
    --dataset_dir "" \
    --train_data_path "" \
    --dev_data_path "" \
    --model_save_dir "" \
    --deepspeed_config ../common/config/deepspeed/dp.conf \
    --use_dp false \
    --lr 4e-5 \
    --micro_batch_size 4 \
    --n_accumulation_steps 16 \
    --n_eval_steps 256 \
    --epoch 3 \
    --is_single True \
    --target_delta 1e-5 \
    --freeze_embedding True \
    --target_epsilon 8 \
    --seed 42
exit 0


##############################################################
# Step2: DP Finetune Generation Proxy model
##############################################################
cd ../common/PrivLM_Bench/examples
deepspeed  \
    --include localhost:2 \
    --master_port 29505  ./llama_finetune.py \
    --model_dir "$Qwen2d5_0d5B_dir" \
    --dataset_name opc_dataset.SingleR_Dataset \
    --collate_fn_name "collate_fn" \
    --dataset_dir "" \
    --train_data_path "" \
    --dev_data_path "" \
    --model_save_dir "" \
    --deepspeed_config ../common/config/deepspeed/dp.conf \
    --use_dp True \
    --lr 4e-5 \
    --micro_batch_size 4 \
    --n_accumulation_steps 16 \
    --n_eval_steps 256 \
    --epoch 3 \
    --is_single True \
    --target_delta 1e-5 \
    --freeze_embedding True \
    --target_epsilon 8 \
    --seed 42
exit 0





##############################################################
# Step3: Construct Training data for Reward Proxy Model
##############################################################
cd ../Medical_QA
finetune_model_dir=""
dp_finetune_model_dir=""
CUDA_VISIBLE_DEVICES=0 python Judge_gen_train_set.py \
    --initial_model_dir "$Qwen2d5_0d5B_dir" \
    --finetune_model_dir "$finetune_model_dir" \
    --dp_finetune_model_dir "$dp_finetune_model_dir" \
    --train_data_path "" \
    --test_data_path "" \
    --dev_data_path "" \
    --batch_size 32 \
    --max_gen_len 512 \
    --output_save_dir "" \
    --gpu_memory_utilization 0.3 \
    --mode "train"

CUDA_VISIBLE_DEVICES=0 python Judge_gen_train_set.py \
    --initial_model_dir "$Qwen2d5_0d5B_dir" \
    --finetune_model_dir "$finetune_model_dir" \
    --dp_finetune_model_dir "$dp_finetune_model_dir" \
    --train_data_path "" \
    --test_data_path "" \
    --dev_data_path "" \
    --batch_size 32 \
    --max_gen_len 512 \
    --output_save_dir "" \
    --gpu_memory_utilization 0.3 \
    --mode "dev"
exit 0


##############################################################
# Step4: DP Finetune Reward Proxy model
##############################################################
cd ../common/PrivLM_Bench/examples
deepspeed  \
    --include localhost:0 \
    --master_port 29507  ./judge_finetune.py \
    --model_dir "$Qwen2d5_0d5B_dir" \
    --dataset_name opc_judge_dataset.Binary_TrainDataset \
    --train_data_path "" \
    --dev_data_path "" \
    --collate_fn_name lambda_collate_fn \
    --model_save_dir "" \
    --deepspeed_config ../common/config/deepspeed/dp.conf \
    --use_dp true \
    --target_epsilon 8 \
    --target_delta 1e-5 \
    --lr 4e-5 \
    --micro_batch_size 4 \
    --n_accumulation_steps 32 \
    --n_eval_steps 2048 \
    --epoch 3 
exit 0
