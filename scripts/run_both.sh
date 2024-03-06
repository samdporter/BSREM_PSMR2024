#!/bin/bash

ALPHAS=(16 32 64 128 256 512 1024 2048 4096)
BETAS=(1 2 4 8 16 32 64 128 256)

PYTHON_SCRIPT="/home/sam/working/BSREM_PSMR2024/main.py"

# run script for each combination of beta and alpha

for beta in "${BETAS[@]}"
do
    for alpha in "${ALPHAS[@]}"
    do
        output_path=results/both/beta_$beta_alpha_$alpha
        mkdir -p $output_dir
        echo "Running with beta=$beta and alpha=$alpha"
        python3 $PYTHON_SCRIPT --beta $beta --alpha $alpha\
                --max_iter=250 --update_interval=10
    done
done

