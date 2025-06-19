# prepare data

I mainly conduct experiment on the COCO dataset, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA) for the file organization. You can download the original dataset from [huggingface](https://huggingface.co/datasets/detection-datasets/coco).

Then extract the images from COCO and save them to ./playground/data/coco/train2017.

# Run the code

```bash
CUDA_VISIBLE_DEVICES=0 python llava/train/train.py
--model_name_or_path liuhaotian/llava-v1.5-7b
--version v1
--data_path /home/Qitao/project/VLM-quant/sharegpt4v_instruct_gpt4-vision_cap100k.json
--image_folder /home/Qitao/project/LLaVA-main/playground/data
--vision_tower openai/clip-vit-large-patch14-336
--mm_projector_type mlp2x_gelu
--mm_vision_select_layer -2
--mm_use_im_start_end False
--mm_use_im_patch_token False
--image_aspect_ratio pad
--group_by_modality_length True
--bf16 True
--output_dir ./checkpoints/llava-v1.5-7b
--num_train_epochs 1
--per_device_train_batch_size 16
--per_device_eval_batch_size 4
--gradient_accumulation_steps 1
--evaluation_strategy "no"
--save_strategy "steps"
--save_steps 50000
--save_total_limit 1
--learning_rate 1e-6
--weight_decay 0.
--warmup_ratio 0.03
--lr_scheduler_type "cosine"
--logging_steps 1
--tf32 True
--model_max_length 2048
--gradient_checkpointing True
--dataloader_num_workers 0
--lazy_preprocess True
```


If you want to try different task, like caption or VQA, you just need to edit the ```--data_path```, 'sharegpt4v_instruct_gpt4-vision_cap100k.json' is a annotation file released by Prismatic VLMs, it's mainly for image captioning task, you can download it [here](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json); 'llava_v1_5_mix665k.json' is another annotation file released by LLaVA, it contains different task for different datasets, and the annotation for COCO in this json file is mainly for VQA, you can down load it [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json). In short, for image captioning task, you set ```--data_path=sharegpt4v_instruct_gpt4-vision_cap100k.json```, for VQA task, set ```--data_path=llava_v1_5_mix665k.json```.


# Details

Use FO or ZO for training, depending on which trainer you use. The ZO trainer is at /llava/train/llava_zo_trainer.py. If you want to use the FO trainer, you need to edit it in /llava/train/train.py.
