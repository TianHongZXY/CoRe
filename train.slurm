#!/bin/bash
  
#SBATCH -J reasoning_math
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:1
#SBATCH -x dgx[050,025,028]
#SBATCH -o logs/gpt2-gsm8k-%j.log
#SBATCH -e errs/gpt2-gsm8k-%j.err
export MASTER_PORT=$[RANDOM%10000+50000]
NUM_NODES=1
NUM_GPUS=1

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=CoRe

ZERO_STAGE=2

config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"
# config_json="./ds_config.json"

cat <<EOT > $config_json
{
    "train_micro_batch_size_per_gpu":2,
    "steps_per_print":10,
    "zero_optimization":{
        "stage": $ZERO_STAGE,
        "offload_optimizer":{
          "device":"cpu",
          "pin_memory":true
        },
        "overlap_comm":true,
        "contiguous_gradients":true,
        "sub_group_size":1000000000,
        "stage3_max_live_parameters":1000000000,
        "stage3_max_reuse_distance":1000000000,
        "stage3_gather_fp16_weights_on_model_save":true
    },
    "zero_allow_untested_optimizer":false,
    "fp16":{
        "enabled":true,
        "loss_scale":0,
        "loss_scale_window":1000,
        "initial_scale_power": 16,
        "hysteresis":2,
        "min_loss_scale":1
    },
    "activation_checkpointing":{
        "partition_activations":false,
        "contiguous_memory_optimization":false
    },
    "wall_clock_breakdown":false
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$config_json

TRAINER_ARGS="
    --max_epochs 10 \
    --gpus $NUM_GPUS \
    --log_every_n_steps 1 \
    --precision 16 \
    --save_dir $ROOT_DIR/outputs \
    --save_top_k 5 \
    --monitor avg_val_loss \
    --mode min \
    --timestamp $TIMESTAMP \
    --gradient_clip_val 1.0 \
    --train \
    --num_nodes $NUM_NODES \
    --accumulate_grad_batches 1 \
    --strategy ddp \
"
    # --strategy deepspeed_stage_2 \
    # --predict \
    # --val_check_interval 0.5 \
    # --patience 3 \
    # --check_val_every_n_epoch 1 \

DATA_DIR=$ROOT_DIR/data/
DATA_ARGS="
    --data_dir $DATA_DIR \
    --num_workers 32 \
    --train_data train_with_qid.jsonl \
    --valid_data test_with_qid.jsonl \
    --test_data test_with_qid.jsonl \
    --micro_batch_size 32 \
    --valid_batch_size 32 \
    --test_batch_size 32 \
    --task gsm8k \
    --recreate_dataset \
    --source_max_token_len 512 \
    --target_max_token_len 512 \
    --data_name gsm8k \
"
    # --predict_batch_size 12 \
    # --predict_data test.jsonl \

MODEL_ARGS="
    --seed 19990303 \
    --model_type gpt \
    --model_name gpt-j \
    --comment gpt-j \
    --lr 1e-5 \
    --l2 0. \
    --warmup 0.1 \
    --show_training_ex 100 \
    --scheduler linear \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
"


SCRIPTS_PATH=$ROOT_DIR/gpt_training_gsm8k.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
bash -c 'python $CMD'
echo "END TIME: $(date)"

