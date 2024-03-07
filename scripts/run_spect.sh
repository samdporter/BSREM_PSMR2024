#!/bin/bash

BETAS=(1 2 4 8 16 32 64 128 256)

PYTHON_SCRIPT="/home/sam/working/BSREM_PSMR2024/main.py"

for beta in ${BETAS[@]}; do
    echo "Running PET with beta = $beta"
    output_path=results/spect/beta_$beta
    mkdir -p $output_path
    python3 $PYTHON_SCRIPT \
            --modality=spect --beta=$beta --output_path=$output_path \
            --max_iter=250 --update_interval=10
done