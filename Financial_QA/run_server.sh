set -e


Qwen2d5_0d5B_dir=""
Qwen2d5_7B_dir=""

cuda_id=0
seed=42
Port=29442
output_save_dir=""
best_rl_path="$output_save_dir/best_rl.pkl"
k_fold=8
n_candidate=6
total_epoch=3
max_gen_len=768
dev_data_path=""
Reward_Proxy_Model=""
Generation_Proxy_Model=""


##############################################################
# Step1: Sampling Synthetic Data
##############################################################
dev_dataset_path=""
CUDA_VISIBLE_DEVICES=$cuda_id python generate_ins.py \
                --finetune_model_dir "$Generation_Proxy_Model" \
                --init_model_dir "$Qwen2d5_0d5B_dir" \
                --dataset_path "$dev_dataset_path"\
                --gen_cnt 6 \
                --max_gen_len 512 \
                --gpu_memory_utilization 0.8 \
                --output_save_dir "$output_save_dir" \
                --target_dataset_size 2994 \
                --seed 42
Question_path="$output_save_dir/gen_Q.json"

CUDA_VISIBLE_DEVICES=$cuda_id python Main_sample_answer.py \
    --finetune_model_dir "$Generation_Proxy_Model" \
    --init_model_dir "$Qwen2d5_0d5B_dir" \
    --dataset_path "$Question_path" \
    --batch_size 32 \
    --gen_cnt $n_candidate \
    --max_gen_len $max_gen_len \
    --gpu_memory_utilization 0.8 \
    --output_save_dir "$output_save_dir" \
    --seed $seed \
    --epoch 1
allTrain_withR_path="$output_save_dir/AllTrain_Generation_Epoch1.json"

##############################################################
# Step2: Reward Guided Filtering
#############################################################
CUDA_VISIBLE_DEVICES=$cuda_id python Main_judge_data.py \
                --initial_model_dir "$Qwen2d5_0d5B_dir" \
                --finetune_model_dir "$Reward_Proxy_Model" \
                --ppl_model_dir "$Generation_Proxy_Model" \
                --dataset_path "$allTrain_withR_path" \
                --output_save_dir $output_save_dir \
                --batch_size 32 \
                --seed $seed \
                --epoch $epoch
All_train_data_path_withJudge="$output_save_dir/AllTrain_Judge_Epoch$epoch.json"

CUDA_VISIBLE_DEVICES=$cuda_id python Main_pre_filter.py \
        --data_path "$All_train_data_path_withJudge" \
        --output_save_dir "$output_save_dir" \
        --total_split $k_fold \
        --seed $seed 
All_select_train_data_path="$output_save_dir/Revise_data_Epoch1.json" #* 原始数据集，不断更新pos_response
Filter_select_train_data_path="$output_save_dir/Filter_revise_data_Epoch1.json" #* 不断扩充的训练池子


##########################################################################################
# Step3: Collabaration of Self-Optimizing Refinement and Reward Guided Filtering
##########################################################################################
deepspeed  \
        --include localhost:$cuda_id \
        --master_port $Port  ./Main_finetune.py \
        --init_model_dir "$Qwen2d5_7B_dir" \
        --model_dir "$Qwen2d5_7B_dir" \
        --dataset_name finqa_dataset.SingleR_Dataset \
        --dataset_dir "" \
        --train_data_path "$Filter_select_train_data_path" \
        --dev_data_path "$dev_data_path" \
        --collate_fn_name "collate_fn" \
        --model_save_dir $output_save_dir \
        --deepspeed_config ../common/config/deepspeed/dp.conf \
        --lr 4e-5 \
        --micro_batch_size 1 \
        --n_accumulation_steps 64 \
        --n_eval_steps 2048 \
        --epoch $k_fold \
        --current_epoch 1 \
        --use_dp false \
        --freeze_embedding True \
        --best_rl_path $best_rl_path \
        --seed $seed \
        --max_generation_len $max_gen_len
Main_model_dir="$output_save_dir/MAIN_Epoch$epoch"

for epoch in $(seq 2 1 $total_epoch)
do
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "Epoch = $epoch"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

        CUDA_VISIBLE_DEVICES=$cuda_id python Main_sample_answer_grad.py \
                --finetune_model_dir "$Generation_Proxy_Model" \
                --init_model_dir "$Qwen2d5_0d5B_dir" \
                --init_opti_dir "$Qwen2d5_7B_dir" \
                --finetune_opti_dir "$Main_model_dir" \
                --dataset_path "$All_select_train_data_path" \
                --batch_size 32 \
                --gen_cnt $n_candidate \
                --max_gen_len $max_gen_len \
                --gpu_memory_utilization 0.8 \
                --output_save_dir "$output_save_dir" \
                --seed $seed \
                --epoch $epoch
        All_Feedback_train_data_path="$output_save_dir/AllTrain_FeedBack_Epoch${epoch}.json"
        All_Revise_train_data_path="$output_save_dir/AllTrain_Revise_Epoch${epoch}.json"


        CUDA_VISIBLE_DEVICES=$cuda_id python Main_judge_data.py \
                --initial_model_dir "$Qwen2d5_0d5B_dir" \
                --finetune_model_dir "$Reward_Proxy_Model" \
                --ppl_model_dir "$Generation_Proxy_Model" \
                --dataset_path "$All_Revise_train_data_path" \
                --output_save_dir $output_save_dir \
                --batch_size 32 \
                --seed $seed \
                --epoch $epoch
        All_train_data_path_withJudge="$output_save_dir/AllTrain_Judge_Epoch$epoch.json"


        CUDA_VISIBLE_DEVICES=$cuda_id python revise_filter.py \
                --data_path "$All_train_data_path_withJudge" \
                --output_save_dir "$output_save_dir" \
                --epoch $epoch \
                --previous_data_path "$Filter_select_train_data_path" \
                --seed $seed \
                --alpha $alpha \
                --diff_thres -1.0 \
                --ratio_to_first 1.0
        All_select_train_data_path="$output_save_dir/Revise_data_Epoch${epoch}.json"
        Filter_select_train_data_path="$output_save_dir/Filter_revise_data_Epoch${epoch}.json"

        deepspeed  \
            --include localhost:$cuda_id \
            --master_port $Port  ./Main_finetune.py \
            --init_model_dir "$Qwen2d5_7B_dir" \
            --model_dir "$Main_model_dir" \
            --dataset_name finqa_dataset.SingleR_Dataset \
            --dataset_dir "" \
            --train_data_path "$Filter_select_train_data_path" \
            --dev_data_path "$dev_data_path" \
            --collate_fn_name "collate_fn" \
            --model_save_dir $output_save_dir \
            --deepspeed_config ../common/config/deepspeed/dp.conf \
            --lr 4e-5 \
            --micro_batch_size 1 \
            --n_accumulation_steps 64 \
            --n_eval_steps 2048 \
            --epoch 3 \
            --current_epoch $epoch \
            --use_dp false \
            --freeze_embedding True \
            --best_rl_path $best_rl_path \
            --seed $seed \
            --max_generation_len $max_gen_len
        Main_model_dir="$output_save_dir/MAIN_Epoch$epoch"

done