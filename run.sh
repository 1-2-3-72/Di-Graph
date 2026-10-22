#!/bin/bash


#UCEC
CUDA_VISIBLE_DEVICES=1 python main.py \
  --data_root_dir /home/oip_3/codes/lxy/TCGA/TCGA-UCEC \
  --split_dir tcga_ucec \
  --lr 2e-4\

#GBMLGG
CUDA_VISIBLE_DEVICES=1 python main.py \
  --data_root_dir /home/oip_3/codes/lxy/TCGA/TCGA-GBMLGG \
  --split_dir tcga_gbmlgg \
  --lr 2e-4\

#BRCA
CUDA_VISIBLE_DEVICES=1 python main.py \
  --data_root_dir /home/oip_3/codes/lxy/TCGA/TCGA-BRCA \
  --split_dir tcga_brca \
  --lr 2e-4\


#BLCA
CUDA_VISIBLE_DEVICES=1 python main.py \
  --data_root_dir /home/oip_3/codes/lxy/TCGA/TCGA-BLCA \
  --split_dir tcga_blca \
  --lr 2e-4\

#LUAD
CUDA_VISIBLE_DEVICES=1 python main.py \
  --data_root_dir /home/oip_3/codes/lxy/TCGA/TCGA-LUAD \
  --split_dir tcga_luad \
  --lr 2e-4\

