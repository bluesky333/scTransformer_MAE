#!/bin/bash

#SBATCH --mail-user jie.sheng@yale.edu
#SBATCH --job-name=submission
#SBATCH --output=log.txt
#SBATCH -p pi_dijk
#SBATCH --mail-type=ALL
#SBATCH --time=5:00
#SBATCH --cpus-per-task=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

python /home/js4435/project/scmae_01_Zhengmix8eq/code/submitit_pretrain.py \
    --job_dir '' \
    --nodes 1 \
    --ngpus 2 \
    --timeout 10080 \
    --batch_size 4 \
    --model mae_vit_d64 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 100 \
    --blr 0.5e-6 --weight_decay 0.05 \
    --file_type 'h5ad' \
    --h5ad_path '/home/js4435/scratch60/K562_gwps_normalized_singlecell_01.h5ad' \
    --label_name gene \
    --partition pi_dijk \
    --partition pi_dijk \
