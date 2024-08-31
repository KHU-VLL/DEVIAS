# Transfer Learning to Downstream Datasets

We initialize the model with weights pre-trained using our DEVIAS training strategy on the source dataset, Kinetics-400, before training on downstream datasets.
Please refer to [TRAIN.md](docs/TRAIN.md) for instructions on how to train DEVIAS.  
We employ various downstream datasets; [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html), [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something), [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php), [ActivityNet](http://activity-net.org/download.html). We use the subset of SSV2 as mentioned in the supplementary of our paper. 

## Fine-tune on Diving48, Something-Something V2, ActivityNet

```bash
# for diving48
OUTPUT_DIR='YOUR_PATH/work_dir/downstream_diving48'
DATA_PATH='YOUR_PATH/filelist/diving48'
DATA_PREFIX='YOUR_PATH/Diving48/rgb'
DATASET='Diving-48'
NUM_CLASSES=48

# for ssv2
OUTPUT_DIR='YOUR_PATH/work_dir/downstream_ssv2'
DATA_PATH='YOUR_PATH/filelist/mini_ssv2'
DATA_PREFIX='YOUR_PATH/something-something-v2'
DATASET='SSV2'
NUM_CLASSES=87

# for activitynet
OUTPUT_DIR='YOUR_PATH/work_dir/downstream_activitynet'
DATA_PATH='YOUR_PATH/filelist/activitynet'
DATA_PREFIX='YOUR_PATH/Activity_256/videos_256'
DATASET='ActivityNet'
NUM_CLASSES=200

MODEL_PATH='YOUR_PATH/ckpt/devias_k400_weights.pth'

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    run_slot_downstream.py \
    --model slot_fusion_vit_base_patch16_224 \
    --data_set $DATASET \
    --downstream_nb_classes $NUM_CLASSES \
    --nb_classes 400 \
    --data_prefix $DATA_PREFIX \
    --data_path $DATA_PATH \
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
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --num_latents 2 \
    --batch_size 12 \
    --head_type 'mlp' \
    --slot_fusion 'concat' \
    --mixup 0.0 \
    --cutmix 0.0 \
    --reprob 0.0 \
    --num_workers 8 \
    --num_latents 2 \
    --agg_block_scale 0.1 \
    --agg_weights_tie --agg_depth 8 
  ```

## Fine-tune on UCF-101

```bash
OUTPUT_DIR='YOUR_PATH/work_dir/downstream_ucf101'
DATA_PATH='YOUR_PATH/filelist/ucf101'
DATA_PREFIX='YOUR_PATH/UCF-101'
DATASET='UCF101'
NUM_CLASSES=101

MODEL_PATH='YOUR_PATH/ckpt/devias_k400_weights.pth'

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    run_slot_downstream.py \
    --model slot_fusion_vit_base_patch16_224 \
    --data_set $DATASET \
    --downstream_nb_classes $NUM_CLASSES \
    --nb_classes 400 \
    --data_prefix $DATA_PREFIX \
    --data_path $DATA_PATH \
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
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --num_latents 2 \
    --batch_size 12 \
    --head_type 'mlp' \
    --slot_fusion 'concat' \
    --mixup 0.0 \
    --cutmix 0.0 \
    --reprob 0.0 \
    --num_workers 8 \
    --num_latents 2 \
    --agg_block_scale 0.1 \
    --agg_weights_tie --agg_depth 8 \
    --fc_drop_rate 0.5 \
    --drop_path 0.2 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --use_input_ln
  ```