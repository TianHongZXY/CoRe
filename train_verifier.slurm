#!/bin/bash
  
#SBATCH -J deberta_verifier
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:1
#SBATCH -o logs/verifier_math-deberta-%j.log
#SBATCH -e errs/verifier_math-deberta-%j.err

export MASTER_PORT=$[RANDOM%10000+50000]
NUM_NODES=1
NUM_GPUS=1

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=acl_supplementary

ZERO_STAGE=2

config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"
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
    --max_epochs 5 \
    --gpus $NUM_GPUS \
    --log_every_n_steps 1 \
    --precision 16 \
    --save_dir $ROOT_DIR/verifier_outputs \
    --save_top_k 5 \
    --monitor avg_train_loss \
    --mode min \
    --timestamp $TIMESTAMP \
    --gradient_clip_val 1.0 \
    --train \
    --num_nodes $NUM_NODES \
    --accumulate_grad_batches 16 \
    --strategy ddp \
"
    # --strategy deepspeed_stage_2 \
    # --predict \
    # --val_check_interval 0.5 \
    # --patience 3 \
    # --check_val_every_n_epoch 1 \

DATA_DIR=$ROOT_DIR/data
DATA_ARGS="
    --data_dir $DATA_DIR \
    --num_workers 32 \
    --train_data examples_for_training_verifiers.jsonl \
    --micro_batch_size 8 \
    --task verifier \
    --recreate_dataset \
"

MODEL_ARGS="
    --seed 19990303 \
    --verifier_loss MSE \
    --model_type deberta \
    --model_name microsoft/deberta-v3-large \
    --lr 1e-5 \
    --l2 0. \
    --warmup 0.1 \
    --show_training_ex 100 \
    --scheduler linear \
"
    # --lm_objective \
    # --model_type gpt \


SCRIPTS_PATH=$ROOT_DIR/verifier_training_gsm8k.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
bash -c 'python $CMD'
echo "END TIME: $(date)"


