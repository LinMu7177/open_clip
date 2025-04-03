#!/bin/bash

echo "Starting process..."

# 等待2小时
sleep 7200

echo "Continuing after 2 hours of sleep..."

# 执行其他命令
nohup python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --dataset-type multi_webdataset \
    --warmup 10000 \
    --batch-size 32 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 10 \
    --model ViT-B-32 \
    --pretrained "/mnt/shared/models/open_clip/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin" \
    --objects-sense-format edges \
    --vl_negs \
    --neg_type "rule" \
    --num_negs 3 \
    --dataset_info "/home/yifei/code/open_clip/datasets_info.yaml" \
    --train_num_workers 12 \
    --val_num_workers 12 \
    > test.log 2>&1 &

echo "Process finished."
