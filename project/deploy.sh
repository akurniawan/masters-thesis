#!/bin/bash

# rsync -r dataset/iwslt14 akurniawan@metacentrum:/storage/plzen1/home/akurniawan/adapters-project/
WORKDIR=/storage/plzen1/home/akurniawan
# WORKDIR=/storage/brno3-cerit/home/akurniawan
# rsync -r dataset/prep_iwslt_wmt.py akurniawan@metacentrum:$WORKDIR/adapters-project/dataset
rsync -r {bin,requirements.txt} akurniawan@metacentrum:$WORKDIR/adapters-project/
rsync experiments_mlm.py akurniawan@metacentrum:$WORKDIR/adapters-project/
rsync experiments_mt.py akurniawan@metacentrum:$WORKDIR/adapters-project/