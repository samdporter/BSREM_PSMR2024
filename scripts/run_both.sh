#!/bin/bash

#ALPHAS=(0 16 32 64 96 128 192 256 512 1024 2048 4096)
#BETAS=(0 0.03125 0.0625 0.125 0.25 0.5 0.75 1 2 4 8 16)

echo "Running both modality with alpha and beta sweep"

Alphas=(128 2048)
BETAS=(0.5 8)

PYTHON_SCRIPT="/home/sam/working/BSREM_PSMR2024/main.py"
base_dir="/home/sam/working/BSREM_PSMR2024/results/both/ab_sweep/mm_test"

# run script for each combination of beta and alpha

for beta in "${BETAS[@]}"
do
    for alpha in "${Alphas[@]}"
    do
        output_path="${base_dir}/alpha_${alpha}_beta_${beta}"
        mkdir -p $output_path
        echo "Running with beta=$beta and alpha=$alpha"
        echo "Output path: $output_path"
        echo "Executing: python3 $PYTHON_SCRIPT --modality=both --beta $beta --alpha $alpha --delta=0.0001 --output_path=$output_path --iterations=1440 --update_interval=72 --relaxation_eta=0.05 --num_subsets=36,24 --gpu --keep_all_views_in_cache --stochastic --svrg --single_modality_update --prior_is_subset"
        python3 $PYTHON_SCRIPT --modality=both --beta $beta --alpha $alpha --delta=0.0001 --output_path=$output_path \
                --iterations=1440 --update_interval=72 --relaxation_eta=0.05 --num_subsets=36,24 \
                --gpu --keep_all_views_in_cache --stochastic --svrg --single_modality_update --prior_is_subset \
                2>&1 | tee -a $output_path/log.txt
        echo "Finished running with beta=$beta and alpha=$alpha"
    done
done
