#!/bin/bash

###########################
### All-purpose testing ###
###########################

# LR_SCHEDULER_GAMMA=0.63  # 0.63^5 = 0.1
# LR_SCHEDULER_GAMMA=0.8  # 0.8^10 = 0.1

EPOCHS=200

DATASET=gtzan_augmented_256_x10_v2
LR=1e-5
TAG=effnet_b1_augmented
ORIGINAL_ONLY_VAL=True
BATCH_SIZE=16 # efficientnetb{0, 1}
LR_SCHEDULER_GAMMA=0.95

# DATASET=gtzan_augmented_256_x0_v3
# LR=1e-4
# TAG=effnet_b1_non_aug_folds
# ORIGINAL_ONLY_VAL=True
# BATCH_SIZE=16 # efficientnetb{0, 1}
# LR_SCHEDULER_GAMMA=0.95

# DATASET=gtzan_augmented_512_x0_v3
# LR=1e-4
# TAG=effnet_b1_non_aug_folds_mel512
# ORIGINAL_ONLY_VAL=True
# BATCH_SIZE=8 # efficientnetb{0, 1}
# LR_SCHEDULER_GAMMA=0.95

# FOLDS="fold_0" # For testing a single fold
FOLDS="fold_0 fold_1 fold_2 fold_3 fold_4" # all 5 folds

# Add timestamp to tag
TAG="$(date +%Y%m%d_%H%M%S)_${TAG}"


for fold in $FOLDS; do
    poetry run \
        python scripts/train.py with \
            dataset=${DATASET} \
            original_only_val=${ORIGINAL_ONLY_VAL} \
            epochs=${EPOCHS} \
            lr=${LR} \
            tag=${TAG} \
            lr_scheduler_gamma=${LR_SCHEDULER_GAMMA} \
            batch_size=${BATCH_SIZE} \
            test_subset=${fold}
            
done