#!/bin/sh
#PBS -N akurniawan_machine_translation
#PBS -l select=1:ncpus=8:ngpus=1:cluster=adan
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -j oe
#PBS -m ae
#PBD -M akurniawan.cs@gmail.com
#PBS -q gpu@cerit-pbs.cerit-sc.cz

PYTHON=/storage/brno3-cerit/home/akurniawan/conda_env/bin/python
WORKDIR=/storage/plzen1/home/akurniawan/adapters-project
# MODELDIR=$WORKDIR/outputs/adapters_de_en
# MODELDIR=$WORKDIR/outputs/baseline_normal
# MODELDIR=$WORKDIR/outputs/adapters_de_en_pt
MODELDIR=$WORKDIR/outputs/baseline_normal_500000.bak
# MODELDIR=$WORKDIR/outputs/baseline_normal_500000
WANDB_API_KEY=45ca700bd4648136b44820bfc7dfde28e203d204
WANDB_PROJECT=iwslt_mt_adapters

# CHECKPOINTDIR=$MODELDIR/checkpoint-200000
# CHECKPOINTDIR=$MODELDIR/checkpoint-120000
# CHECKPOINTDIR=$MODELDIR/checkpoint-234000
CHECKPOINTDIR=$MODELDIR/checkpoint-214000
# CHECKPOINTDIR=$MODELDIR/checkpoint-200000
CUDA_VISIBLE_DEVICES=1 $PYTHON $WORKDIR/experiments_mt.py \
    --seq2seq_model_path $CHECKPOINTDIR \
    --enc_config_name bert-base-german-dbmdz-uncased \
    --dec_config_name bert-base-uncased \
    --dataset_loader_script $WORKDIR/dataset/iwslt14/iwslt_loader.py \
    --dataset_config_name de-en \
    --source_lang de \
    --target_lang en \
    --dataset_dir $WORKDIR/dataset/iwslt14 \
    --output_dir $MODELDIR \
    --max_source_length 512 \
    --preprocessing_num_workers 8 \
    --pad_to_max_length True \
    --per_gpu_eval_batch_size 16 \
    --num_beams 5 --do_eval --do_predict --predict_with_generate

# CUDA_VISIBLE_DEVICES=3 $PYTHON $WORKDIR/experiments_mt.py \
#     --seq2seq_model_path $CHECKPOINTDIR \
#     --enc_config_name bert-base-german-dbmdz-uncased \
#     --dec_config_name bert-base-uncased \
#     --enc_adapters_name iwslt_adapters \
#     --dec_adapters_name iwslt_adapters \
#     --dataset_loader_script $WORKDIR/dataset/iwslt14/iwslt_loader.py \
#     --dataset_config_name de-en \
#     --source_lang de \
#     --target_lang en \
#     --dataset_dir $WORKDIR/dataset/iwslt14 \
#     --output_dir $WORKDIR/outputs/adapters_models \
#     --max_source_length 512 \
#     --preprocessing_num_workers 8 \
#     --pad_to_max_length True \
#     --per_gpu_eval_batch_size 64 \
#     --num_beams 5 --do_eval --do_predict --predict_with_generate
