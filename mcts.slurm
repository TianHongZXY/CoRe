#!/bin/bash
  
#SBATCH -J mcts_test
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:1
#SBATCH -x dgx[045]
#SBATCH -o logs/mcts-gsm8k-test-%j.log
#SBATCH -e errs/mcts-gsm8k-test-%j.err

NUM_NODES=1
NUM_GPUS=1

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=acl_supplementary


MCTS_ARGS="
    --max_num_children 2 \
    --root_max_num_children 20 \
    --roll_out_size 300 \
    --sampling_size 50 \
    --max_length 400 \
    --max_iter 40 \
    --sim_score_base 0.6 \
    --time_out 320 \
    --alpha 1. \
    --sample_capacity 1000 \
    --temperature 0.7 \
    --data $ROOT_DIR/data/test_with_qid.jsonl \
    --model_name gpt2 \
    --verifier_type deberta \
    --verifier_name microsoft/deberta-v3-large \
    --expand_verifier_type gpt \
    --expand_verifier_name gpt2 \
    --timestamp $TIMESTAMP \
    --data_name gsm8k_test \
    --expand_length 20 \
    --split 0 \
    --expand_repeat_penalty 1.2 \
"


SCRIPTS_PATH=$ROOT_DIR/mcts.py

export CMD=" \
    $SCRIPTS_PATH \
    $MCTS_ARGS \
    "

echo $CMD
bash -c 'python $CMD'
echo "END TIME: $(date)"

