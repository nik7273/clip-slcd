#!/bin/bash


UHUMAN_PATH=/home/nikhil/Documents/rob_530/clip-slcd/datasets/uhumans

evalset=(
    left_images
)

for seq in ${evalset[@]}; do
    echo $seq
    python evaluation_scripts/test_uhuman.py --datapath=$UHUMAN_PATH/$seq --weights=droid.pth --reconstruction_path=uhuman_apartment $@
done

