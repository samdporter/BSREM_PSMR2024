#!/bin/bash

#ALPHAS=(8 16 32 64 128 256 512) # incorrect
#ALPHAS=(32 64 96 128 192 256 512) # correct
ALPHAS=(512 2048 8192 16384 32768 65536 131072 262144 524288 1048576)
beta=1

PYTHON_SCRIPT="/home/sam/working/BSREM_PSMR_MIC_2024/main.py"
base_dir="/home/sam/working/BSREM_PSMR_MIC_2024/results/both/testTNV"

for alpha in ${ALPHAS[@]}; do
    echo "Running PET with alpha = $alpha"
    output_path="${base_dir}/alpha_${alpha}"
    mkdir -p $output_path
    python3 $PYTHON_SCRIPT --modality=both --beta $beta --alpha $alpha --delta=0.0001 --output_path=$output_path \
            --iterations=60 --update_interval=12 --relaxation_eta=0.05 \
            --gpu --keep_all_views_in_cache
done