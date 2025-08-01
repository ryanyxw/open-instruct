#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=/home/ryan/decouple/scripts/slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:2"
#SBATCH --ntasks=16
#SBATCH --exclude=lime-mint,allegro-adams,glamor-ruby

path_to_hf_model="../models/figure2/low-risk_partition0/step31-unsharded/hf"
save_dir="../models/figure2/low-risk_partition0/step31-unsharded/instruction_tuned"
exp_name_="tulu-v2-sft-mixture-low-risk_partition0_sft"

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 2 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    --main_process_port 29502 \
    open_instruct/finetune.py \
    --model_name_or_path ${path_to_hf_model} \
    --use_slow_tokenizer False \
    --use_flash_attn \
    --max_seq_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir ${save_dir} \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-v2-sft-mixture-olmo-2048 150000 \
    --push_to_hub False \
    --try_launch_beaker_eval_jobs False \
    --max_train_samples 150000 \
    --exp_name ${exp_name_} \
    --seed 123 \
    --add_bos