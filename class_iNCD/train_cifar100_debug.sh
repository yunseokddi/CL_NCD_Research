#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 torchrun \
        --nproc_per_node=2 \
        train.py \
        cifar100 \
        --tensorboard False