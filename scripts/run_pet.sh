#!/bin/bash

ALPHAS=(16 32 64 128 256 512 1024 2048 4096)

PYTHON_SCRIPT="/home/sam/working/BSREM_PSMR2024/main.py"

for alpha in ${ALPHAS[@]}; do
    echo "Running PET with alpha = $alpha"
    output_dir=results/pet/alpha_$alpha
    mkdir -p $output_dir
    python3 $PYTHON_SCRIPT \
            --modality=pet --alpha=$alpha --output_path=$output_dir \
            --max_iter=250 --update_interval=10
done