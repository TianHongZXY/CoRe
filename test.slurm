#!/bin/bash

#SBATCH -J predict_reasoning
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:1
#SBATCH -x dgx[050]
#SBATCH -o logs/predict_reasoning_math-gpt-2-%j.log
#SBATCH -e errs/predict_reasoning_math-gpt-2-%j.err

# set -x -e
export MASTER_PORT=$[RANDOM%10000+50000]
NUM_NODES=1
NUM_GPUS=1

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=acl_supplementary

TRAINER_ARGS="
    --gpus $NUM_GPUS \
    --save_dir $ROOT_DIR/outputs \
    --timestamp $TIMESTAMP \
    --precision 16 \
    --predict \
    --sample_len 300 \
    --strategy ddp \
"

DATA_DIR=$ROOT_DIR/data/
DATA_ARGS="
    --data_dir $DATA_DIR \
    --num_workers 16 \
    --predict_data test_with_qid.jsonl \
    --predict_batch_size 64 \
    --recreate_dataset \
    --task predictor \
    --data_name gsm8k \
"

MODEL_ARGS="
    --seed 19990303 \
    --model_type gpt \
    --model_name gpt2 \
"

SCRIPTS_PATH=$ROOT_DIR/gpt_training_gsm8k.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
bash -c 'time python $CMD'
echo "END TIME: $(date)"

