# Adapter Experiments

## Running LM experiment

```python3
python experiments_mlm.py \
    --model_type bert \
    --config_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --train_file dataset/iwslt14/train.en \
    --validation_file dataset/iwslt14/valid.en \
    --output_dir outputs/models \
    --preprocessing_num_workers 8 \
    --do_train --do_eval --overwrite_output_dir \
    --report_to wandb
```

## Running from scratch MT experiment

```python3
python experiments_mt.py \
    --enc_config_name bert-base-uncased \
    --dec_config_name bert-base-german-dbmdz-uncased \
    --dataset_loader_script dataset/iwslt14/iwslt_loader.py \
    --dataset_config_name de-en \
    --source_lang de \
    --target_lang en \
    --dataset_dir dataset/iwslt14 \
    --output_dir outputs/models \
    --max_source_length 512 \
    --preprocessing_num_workers 8 \
    --pad_to_max_length True \
    --do_train --do_eval --overwrite_output_dir \
    --report_to wandb
```

## Running MT experiment with adapters

```python3
python experiments_mt.py \
    --enc_config_name bert-base-uncased \
    --dec_config_name bert-base-german-dbmdz-uncased \
    --enc_adapters_name iwslt_adapters \
    --dec_adapters_name iwslt_adapters \
    --dataset_loader_script dataset/iwslt14/iwslt_loader.py \
    --dataset_config_name de-en \
    --source_lang de \
    --target_lang en \
    --dataset_dir dataset/iwslt14 \
    --output_dir outputs/models \
    --max_source_length 512 \
    --preprocessing_num_workers 8 \
    --pad_to_max_length True \
    --do_train --do_eval --overwrite_output_dir \
    --report_to wandb
```