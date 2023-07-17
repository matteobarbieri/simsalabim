#!/bin/bash

EPOCHS=3
LR=1e-4
# DATASET=gtzan_augmented_256_x10_v2
DATASET=gtzan_processed
BATCH_SIZE=64

# for fold in fold_0 fold_1 fold_2 fold_3 fold_4; do
# for fold in fold_0 fold_1; do
for fold in fold_0; do
    poetry run \
        python scripts/train.py with \
            dataset=${DATASET} \
            epochs=${EPOCHS} \
            lr=${LR} \
            tag=resnet18_focal \
            batch_size=${BATCH_SIZE} \
            test_subset=${fold}
done