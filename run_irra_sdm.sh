#!/bin/bash
DATASET_NAME="CUHK-PEDES"
#DATASET_NAME='ICFG-PEDES'
#DATASET_NAME='RSTPReid'  #lr 1e-4


CUDA_VISIBLE_DEVICES=2 \
python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm' \
--num_epoch 60 \
--root_dir '.../dataset_reid' \
--lr 1e-3 \
--prefix_length 10 \
--rank 32 \
--depth_lora 12 \
--depth_prefix 12 \
--depth_adapter 0 
