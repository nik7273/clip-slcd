#!/bin/bash


TUM_PATH=datasets/TUM-RGBD/$seq

evalset=(
    rgbd_dataset_freiburg1_360
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_tum.py --datapath=$TUM_PATH/$seq --weights=droid.pth$@
done

