# Evaluation DEVIAS

## Prepare datasets  
Please prepare the datasets by following the instructions in [DATASET.md](docs/DATASET.md) before evaluating the models.

## Evaluate the **scene** recongnition performance in *seen* combination scenarios
To evaluate the scene recognition performance on **UCF-101**/**Kinetics-400**, you just need to add ```--eval --eval_scene``` to the arguments used for training.

```bash
MODEL_PATH='YOUR_PATH/logs/devias_k400_weights.pth'
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT run_slot_finetuning.py \
    ... \  #(same as the arguments used in training)
    --finetune $MODEL_PATH \
    --eval --eval_scene
```

## Evaluate the **action** recongnition performance in *unseen* combination scenarios
To evaluate the action recognition performance on **SCUBA**, you need to add ```--run_scuba``` and change ``` --data_prefix $SCUBA_DATA_PREFIX``` to the arguments used for training.

```bash
MODEL_PATH='YOUR_PATH/logs/devias_k400_weights.pth'
SCUBA_DATA_PREFIX='YOUR_PATH/scuba/kinetics-vqgan'
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT run_slot_finetuning.py \
    ... \  #(same as the arguments used in training)
    --finetune $MODEL_PATH \
    --run_scuba --data_prefix $SCUBA_DATA_PREFIX
```

To evaluate the action recognition performance on **HAT-Far/Random/Close**, you need to add ```--hat_eval --hat_anno_path $HAT_ANNO_DATA_PATH``` and change ``` --data_prefix $HAT_DATA_PREFIX``` to the arguments used for training.

```bash
MODEL_PATH='YOUR_PATH/logs/devias_k400_weights.pth'
HAT_DATA_PREFIX='YOUR_PATH/hat/kinetics'
HAT_ANNO_DATA_PATH='YOUR_PATH/filelist/hat/kinetics/far'
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT run_slot_finetuning.py \
    ... \  #(same as the arguments used in training)
    --finetune $MODEL_PATH \
   --hat_eval \
   --hat_anno_path $HAT_ANNO_DATA_PATH \
   --data_prefix $HAT_DATA_PREFIX
```

## Evaluate the **scene** recongnition performance in *unseen* combination scenarios
To evaluate the scene recognition performance on **HAT-Scene-Only**, you need to add ```--eval --eval_scene``` and change ``` --data_prefix $HAT_DATA_PREFIX --data_path $HAT_ANNO_DATA_PATH``` to the arguments used for training.

```bash
MODEL_PATH='YOUR_PATH/logs/devias_k400_weights.pth'
HAT_DATA_PREFIX='YOUR_PATH/hat/kinetics'
HAT_ANNO_DATA_PATH='YOUR_PATH/filelist/hat/kinetics/list.csv'
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT run_slot_finetuning.py \
    ... \  #(same as the arguments used in training)
    --finetune $MODEL_PATH \
   --eval --eval_scene \
   --data_prefix $HAT_DATA_PREFIX --data_path $HAT_ANNO_DATA_PATH
```

To evaluate the scene recognition performance on **HAT-Far/Random/Close**, you need to add ```--hat_eval --eval_scene --hat_anno_path $HAT_ANNO_DATA_PATH``` and change ``` --data_prefix $HAT_DATA_PREFIX``` to the arguments used for training.

```bash
MODEL_PATH='YOUR_PATH/logs/devias_k400_weights.pth'
HAT_DATA_PREFIX='YOUR_PATH/hat/kinetics'
HAT_ANNO_DATA_PATH='YOUR_PATH/filelist/hat/kinetics/far'
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT run_slot_finetuning.py \
    ... \  #(same as the arguments used in training)
    --finetune $MODEL_PATH \
   --hat_eval --eval_scene \
   --hat_anno_path $HAT_ANNO_DATA_PATH \
   --data_prefix $HAT_DATA_PREFIX
```

## Evaluate on HVU 
To evaluate DEVIAS on **HVU**, you need to change ```--data_set HVU-EVAL --anno_path $HVU_SEEN_ANNO_PATH $HVU_UNSEEN_ANNO_PATH``` to the arguments used for training. You can evaluate both action and scene recognition performances through the below command.

```bash
MODEL_PATH='YOUR_PATH/logs/devias_hvu_weights.pth'
HVU_SEEN_ANNO_PATH='YOUR_PATH/filelist/hvu/val_seen.csv'
HVH_UNSEEN_ANNO_PATH='YOUR_PATH/filelist/hvu/val_unseen.csv'
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT eval_slot_finetuning_hvu.py \
    ... \  #(same as the arguments used in training)
    --finetune $MODEL_PATH \
    --data_set HVU-EVAL \
    --anno_path $HVU_SEEN_ANNO_PATH $HVU_UNSEEN_ANNO_PATH
```