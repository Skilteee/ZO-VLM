#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python llava/train/train.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /home/Qitao/project/VLM-quant/sharegpt4v_instruct_gpt4-vision_cap100k.json \
    --image_folder /home/Qitao/project/LLaVA-main/playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

#python llava/train/train.py \
#    --model_name_or_path liuhaotian/llava-v1.5-7b \
#    --version v1 \
#    --data_path /home/Qitao/project/VLM-quant/sharegpt4v_instruct_gpt4-vision_cap100k.json \
#    --image_folder ./playground/data \
#    --vision_tower openai/clip-vit-large-patch14-336 \
#    --mm_projector_type mlp2x_gelu \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --image_aspect_ratio pad \
#    --group_by_modality_length True \
#    --bf16 True \
#    --output_dir ./checkpoints/llava-v1.5-13b-task \
#    --num_train_epochs 100 \
#    --per_device_train_batch_size 1 \
#    --per_device_eval_batch_size 1 \
#    --gradient_accumulation_steps 1 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 50000 \
#    --save_total_limit 1 \
#    --learning_rate 1e-6 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --dataloader_num_workers 0 \
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#    --lazy_preprocess True
