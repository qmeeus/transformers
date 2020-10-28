#!/bin/bash

# set -x
set -e
set -o pipefail

CONDA_HOME=/esat/spchdisk/scratch/qmeeus/bin/anaconda3
CONDA_INIT=$CONDA_HOME/etc/profile.d/conda.sh
CONDA_ENV=pytorch-gpu
source $CONDA_INIT
echo "Activate conda environment $CONDA_ENV"
conda activate $CONDA_ENV

DATA_DIR="/esat/spchdisk/scratch/qmeeus/repos/espnet/egs/cgn/asr1/dump"
TRAIN_FILE="$DATA_DIR/CGN_train/deltafalse/data_unigram_5000.o.json"
EVAL_FILE="$DATA_DIR/CGN_valid/deltafalse/data_unigram_5000.o.json"

[ $(hostname) == "fasso.esat.kuleuven.be" ] && { BATCH_SIZE=2; OUTDIR=debug; } || { BATCH_SIZE=4; OUTDIR=output; }

/bin/env python run_asr.py \
    --output_dir=$OUTDIR \
    --model_type=speechbert \
    --tokenizer_name="tokenizer" \
    --num_train_epochs=100 \
    --load_best_model_at_end \
    --do_train \
    --train_data_file="$TRAIN_FILE" \
    --do_eval \
    --eval_data_file="$EVAL_FILE" \
    --evaluation_strategy=epoch \
    --cache_dir=data \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=8 \
    --overwrite_output_dir


