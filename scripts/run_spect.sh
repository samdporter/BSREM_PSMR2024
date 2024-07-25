#!/bin/bash

alpha=0
BETAS=(0.0625 0.125 0.25 0.5 0.75 1 2)
base_dir="/home/sam/working/BSREM_PSMR2024/results/spect/ab_sweep"

PYTHON_SCRIPT="/home/sam/working/BSREM_PSMR2024/main.py"

for beta in ${BETAS[@]}; do
    echo "Running SPECT with beta = $beta"
    output_path="${base_dir}/beta_${beta}"
    mkdir -p $output_path
    python3 $PYTHON_SCRIPT --modality=spect --beta $beta --alpha $alpha --delta=0.0001 --output_path=$output_path \
            --iterations=600 --update_interval=12 --relaxation_eta=0.05 \
            --gpu --keep_all_views_in_cache
done