#!/bin/sh
#PBS -N akurniawan_machine_translation_adapters
#PBS -l select=1:ncpus=8:ngpus=1:mem=20gb:cluster=galdor
#PBS -j oe
#PBS -m abe
#PBD -M akurniawan.cs@gmail.com
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -N akurniawan_mt_adapter_zsbert_zsbert

PYTHON=/storage/brno3-cerit/home/akurniawan/conda_env/bin/python
WORKDIR=/storage/plzen1/home/akurniawan/adapters-project
MODELDIR=/storage/brno3-cerit/home/akurniawan/adapters-project/outputs
CACHE_FOLDER=/storage/brno3-cerit/home/akurniawan/adapters-project/.cache

WANDB_API_KEY=45ca700bd4648136b44820bfc7dfde28e203d204 \
WANDB_PROJECT=iwslt_adapter_zsbert_zsbert \
$PYTHON $WORKDIR/experiments_mt.py \
    --enc_config_name bert-base-german-dbmdz-uncased \
    --dec_config_name bert-base-uncased \
    --enc_adapters_name iwslt_adapters \
    --enc_model_name_or_path /storage/brno3-cerit/home/akurniawan/adapters-project/outputs/bert_de_zero_reduced \
    --dec_model_name_or_path /storage/brno3-cerit/home/akurniawan/adapters-project/outputs/bert_en_zero_reduced \
    --dataset_loader_script $WORKDIR/dataset/iwslt14/iwslt_loader.py \
    --dataset_dir $WORKDIR/dataset/iwslt14 \
    --dataset_config_name de-en \
    --source_lang de \
    --target_lang en \
    --output_dir $MODELDIR/adapter_zsbert_zsbert \
    --max_source_length 512 \
    --max_target_length 180 \
    --preprocessing_num_workers 8 \
    --evaluation_strategy "steps" \
    --load_best_model_at_end True \
    --metric_for_best_model "bleu" \
    --learning_rate 0.0005 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 40.0 \
    --warmup_steps 4000 \
    --pad_to_max_length True \
    --do_train --do_eval --do_predict --predict_with_generate \
    --report_to wandb