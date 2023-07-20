#!/bin/bash

###########################
### All-purpose testing ###
###########################

EPOCHS=200

DATASET=gtzan_augmented_256_x0_no_aug
MODEL_NAME=effnet_b1
LR=1e-4
LR_SCHEDULER_GAMMA=0.95
ORIGINAL_ONLY_VAL=True
BATCH_SIZE=16 # efficientnetb{0, 1}
TAG=no_pretrain_no_aug
PRETRAINED=False

# DATASET=gtzan_augmented_256_x5
# MODEL_NAME=effnet_b1
# LR=5e-5
# LR_SCHEDULER_GAMMA=0.95
# ORIGINAL_ONLY_VAL=True
# BATCH_SIZE=16 # efficientnetb{0, 1}
# TAG=no_pretrain
# PRETRAINED=False


# FOLDS="fold_0" # For testing a single fold
FOLDS="fold_0 fold_1 fold_2 fold_3 fold_4" # all 5 folds

# Add model name and timestamp to tag
TAG="${MODEL_NAME}_${TAG}"
TAG="$(date +%Y%m%d_%H%M%S)_${TAG}"


for fold in $FOLDS; do
    poetry run \
        python scripts/train.py with \
            dataset=${DATASET} \
            original_only_val=${ORIGINAL_ONLY_VAL} \
            model_name=${MODEL_NAME} \
            pretrained=${PRETRAINED} \
            epochs=${EPOCHS} \
            lr=${LR} \
            tag=${TAG} \
            lr_scheduler_gamma=${LR_SCHEDULER_GAMMA} \
            batch_size=${BATCH_SIZE} \
            test_subset=${fold}
            
done
