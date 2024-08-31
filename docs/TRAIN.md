# Training DEVIAS

We initialize our model with weights pre-trained by [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md) on the target dataset.
Please download ViT-B weights pre-trained on Kinetics-400 or ViT-B weights fine-tuned on UCF-101.
(In the case of UCF-101 training, we use fine-tuned weights for stabilizing the training process. Please refer the supplementary of our paper.)
<!-- DEVIAS adopts the training code from [VideoMAE](https://github.com/MCG-NJU/VideoMAE). We appreciate their contributions to the original code. -->

## Train on Kinetics-400

<!-- We fine-tune DEVIAS on Kinetics-400 with 16 RTX3090 GPUs. -->

```bash
NUM_GPUS=8
MASTER_PORT=36524
OUTPUT_DIR='YOUR_PATH/work_dir/train_k400'
DATA_PATH='YOUR_PATH/filelist/k400'
DATA_PREFIX='YOUR_PATH/Kinetics-400'
MODEL_PATH='YOUR_PATH/ckpt/videomae_k400_weights.pth'
SCENE_MODEL_PATH='YOUR_PATH/ckpt/scene_model_ckpt.pth'

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    run_slot_finetuning.py \
    --model slot_vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path $DATA_PATH \
    --data_prefix $DATA_PREFIX \
    --finetune $MODEL_PATH \
    --log_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 50 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 1 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --agg_block_scale 0.1 \
    --slot_matching_method 'matching' \
    --batch_size 12 \
    --head_type 'linear' \
    --mixup 0.0 \
    --cutmix 0.0 \
    --reprob 0.0 \
    --num_workers 8 \
    --mask_model FAME \
    --beta 0.5 \
    --prob_aug 0.8 \
    --mask_distill_loss_weight 1.0 \
    --mask_prediction_loss_weight 1.0 \
    --num_latents 2 \
    --agg_weights_tie --agg_depth 8 \
    --scene_model_path $SCENE_MODEL_PATH
  ```


<!-- If you just want to **test the performance of the model**, change `MODEL_PATH` to the model to be tested, `OUTPUT_DIR` to the path of the folder where the test results are saved, and add the `--eval` argument to the end of the upper script. -->

## Train on UCF-101

<!-- We fine-tune DEVIAS on Kinetics-400 with 16 RTX3090 GPUs. -->

```bash
NUM_GPUS=8
MASTER_PORT=36524
OUTPUT_DIR='YOUR_PATH/work_dir/train_ucf101'
DATA_PATH='YOUR_PATH/filelist/ucf101'
DATA_PREFIX='YOUR_PATH/UCF-101'
MODEL_PATH='YOUR_PATH/ckpt/videomae_ucf101_weights.pth'
SCENE_MODEL_PATH='YOUR_PATH/ckpt/scene_model_ckpt.pth'

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    run_slot_finetuning.py \
    --model slot_vit_base_patch16_224 \
    --data_set UCF101 \
    --nb_classes 101 \
    --data_path $DATA_PATH \
    --data_prefix $DATA_PREFIX \
    --finetune $MODEL_PATH \
    --log_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 50 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 1 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --agg_block_scale 0.1 \
    --slot_matching_method 'matching' \
    --batch_size 12 \
    --head_type 'linear' \
    --mixup 0.0 \
    --cutmix 0.0 \
    --reprob 0.0 \
    --fc_drop_rate 0.5 \
    --drop_path 0.2 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --num_workers 8 \
    --mask_model FAME \
    --beta 0.3 \
    --prob_aug 0.4 \
    --mask_distill_loss_weight 1.0 \
    --mask_prediction_loss_weight 1.0 \
    --num_latents 2 \
    --agg_weights_tie --agg_depth 4 \
    --scene_model_path $SCENE_MODEL_PATH
  ```


## Train on HVU

```bash
NUM_GPUS=8
MASTER_PORT=36524
OUTPUT_DIR='YOUR_PATH/work_dir/train_hvu'
DATA_PATH='YOUR_PATH/filelist/hvu'
DATA_PREFIX='YOUR_PATH/HVU'
MODEL_PATH='YOUR_PATH/ckpt/videomae_400_weights.pth'

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    run_slot_finetuning_hvu.py \
    --model slot_vit_base_patch16_224 \
    --data_set HVU \
    --nb_classes 739 \
    --data_path $DATA_PATH \
    --data_prefix $DATA_PREFIX \
    --finetune $MODEL_PATH \
    --log_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 50 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 1 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --enable_deepspeed \
    --agg_block_scale 0.1 \
    --slot_matching_method 'matching' \
    --batch_size 12 \
    --head_type 'linear' \
    --mixup 0.0 \
    --cutmix 0.0 \
    --reprob 0.0 \
    --num_workers 8 \
    --mask_model FAME \
    --beta 0.5 \
    --prob_aug 0.25 \
    --mask_distill_loss_weight 1.0 \
    --mask_prediction_loss_weight 1.0 \
    --num_latents 2 \
    --agg_weights_tie --agg_depth 8 