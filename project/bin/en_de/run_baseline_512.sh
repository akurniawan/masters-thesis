#!/bin/sh
#PBS -l select=1:ncpus=8:ngpus=1:mem=20gb:cluster=zia
#PBS -j oe
#PBS -m abe
#PBD -M akurniawan.cs@gmail.com
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -N akurniawan_mt_baseline_bert_en_de_512

PYTHON=/storage/brno3-cerit/home/akurniawan/conda_env/bin/python
WORKDIR=/storage/plzen1/home/akurniawan/adapters-project
MODELDIR=/storage/brno3-cerit/home/akurniawan/adapters-project/outputs
CACHE_FOLDER=/storage/brno3-cerit/home/akurniawan/adapters-project/.cache

WANDB_API_KEY=45ca700bd4648136b44820bfc7dfde28e203d204 \
WANDB_PROJECT=iwslt_baseline_bert_en_de \
$PYTHON $WORKDIR/experiments_mt.py \
    --dec_config_name bert-base-german-dbmdz-uncased \
    --enc_config_name bert-base-uncased \
    --dec_model_name_or_path bert-base-german-dbmdz-uncased \
    --enc_model_name_or_path bert-base-uncased \
    --dataset_loader_script $WORKDIR/dataset/iwslt14/iwslt_loader.py \
    --dataset_config_name de-en \
    --source_lang en \
    --target_lang de \
    --dataset_dir $WORKDIR/dataset/iwslt14 \
    --output_dir $MODELDIR/baseline_bert_en_de_512 \
    --max_source_length 512 \
    --max_target_length 512 \
    --preprocessing_num_workers 8 \
    --evaluation_strategy "steps" \
    --load_best_model_at_end True \
    --metric_for_best_model "bleu" \
    --learning_rate 0.0001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --max_steps 400600 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 40.0 \
    --warmup_steps 4000 \
    --pad_to_max_length True \
    --do_train --do_eval --predict_with_generate \
    --report_to wandb

