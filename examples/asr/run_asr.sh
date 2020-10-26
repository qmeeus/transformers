#!/bin/bash

# set -x
set -e
set -o pipefail

CONDA_HOME=/home/quent/.local/bin/anaconda3
CONDA_INIT=$CONDA_HOME/etc/profile.d/conda.sh
CONDA_ENV=pytorch-gpu
source $CONDA_INIT
echo "Activate conda environment $CONDA_ENV"
conda activate $CONDA_ENV

DATA_DIR="/run/media/quent/QM HD 1To USB 3.0/data/cgn/features"
TRAIN_FILE="$DATA_DIR/train/data_unigram_5000.o.json"
EVAL_FILE="$DATA_DIR/valid/data_unigram_5000.o.json"

/bin/env python run_asr.py \
    --output_dir=output \
    --model_type=speechbert \
    --tokenizer_name="tokenizer" \
    --do_train \
    --train_data_file="$TRAIN_FILE" \
    --do_eval \
    --eval_data_file="$EVAL_FILE" \
    --eval_steps=500 \
    --cache_dir=data \
    --per_device_train_batch_size=3 \
    --gradient_accumulation_steps=8 \
    --overwrite_output_dir


