#!/bin/bash

ALPHAS=(128 256 512 1024 2048 4096)
BETAS=(0.125 0.25 0.5 1 2 4)

PYTHON_SCRIPT="/home/sam/working/BSREM_PSMR2024/main.py"

# run script for each combination of beta and alpha

for beta in "${BETAS[@]}"
do
    for alpha in "${ALPHAS[@]}"
    do
        output_path=/home/sam/working/BSREM_PSMR2024/results/both/beta_${beta}_alpha_${alpha}
        mkdir -p $output_path
        echo "Running with beta=$beta and alpha=$alpha"
        python3 $PYTHON_SCRIPT --beta $beta --alpha $alpha --output_path=$output_path \
                --iterations=120 --update_interval=6
    done
done

