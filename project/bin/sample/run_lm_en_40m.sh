#!/bin/sh
#PBS -l select=1:ncpus=8:ngpus=1:mem=20gb:cluster=zia
#PBS -j oe
#PBS -m abe
#PBD -M akurniawan.cs@gmail.com
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -N akurniawan_lm_en_40m

PYTHON=/storage/brno3-cerit/home/akurniawan/conda_env/bin/python
WORKDIR=/storage/plzen1/home/akurniawan/adapters-project
API_KEY=45ca700bd4648136b44820bfc7dfde28e203d204
LANG=en

SAMPLE=40000000

WANDB_API_KEY=$API_KEY WANDB_PROJECT=iwslt_lm_$LANG\_$SAMPLE $PYTHON $WORKDIR/experiments_mlm.py \
    --model_type bert \
    --config_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --dataset_disk_path $WORKDIR/dataset/wmt19_sample_$SAMPLE\_$LANG \
    --output_dir $WORKDIR/outputs/iwslt_lm_$LANG\_$SAMPLE \
    --preprocessing_num_workers 8 \
    --evaluation_strategy "steps" \
    --load_best_model_at_end True \
    --learning_rate 0.0001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 500.0 \
    --warmup_steps 10000 \
    --do_train --do_eval \
    --report_to wandb
